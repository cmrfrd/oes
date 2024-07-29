#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::clip;

use image;
use tokenizers::Tokenizer;
use tracing::info;

use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Specifies a model file
    #[clap(long)]
    model: Option<String>,

    /// Specifies a tokenizer file
    #[clap(long)]
    tokenizer: Option<String>,

    /// List of images
    #[clap(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    /// Use CPU for processing
    #[clap(long)]
    cpu: bool,

    /// List of sequences
    #[clap(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    // .unsqueeze(0)?;
    Ok(img)
}

fn load_images<T: AsRef<std::path::Path>>(
    paths: &Vec<T>,
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];

    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor);
    }

    let images = Tensor::stack(&images, 0)?;

    Ok(images)
}

pub fn main() -> anyhow::Result<()> {
    // std::env::set_var("RUST_BACKTRACE", "full");

    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;

            let api = api.repo(hf_hub::Repo::with_revision(
                "openai/clip-vit-base-patch32".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/15".to_string(),
            ));

            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    let tokenizer = get_tokenizer(args.tokenizer)?;

    let config = clip::ClipConfig::vit_base_patch32();

    let device = device(args.cpu)?;

    let batch = 16;
    let vec_imgs = match args.images {
        Some(imgs) => imgs,
        None => vec!["./stable-diffusion-xl.jpg".to_string(); batch],
    };

    // let image = load_image(args.image, config.image_size)?.to_device(&device)?;
    let images = load_images(&vec_imgs, config.image_size)?.to_device(&device)?;

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)? };

    let model = clip::ClipModel::new(vb, &config)?;

    let (input_ids, vec_seq) = tokenize_sequences(args.sequences, &tokenizer, &device)?;

    let time_start = std::time::Instant::now();
    let (_logits_per_text, logits_per_image) = model.forward(&images, &input_ids)?;
    let elapsed = time_start.elapsed();
    println!("Elapsed time: {:?}", elapsed);
    println!("Batch size: {:?}", vec_imgs.len() as u32);
    println!("Time per image: {:?}", elapsed / vec_imgs.len() as u32);

    let softmax_image = softmax(&logits_per_image, 1)?;

    let softmax_image_vec = softmax_image.flatten_all()?.to_vec1::<f32>()?;

    // info!("softmax_image_vec: {:?}", softmax_image_vec);

    let probability_vec = softmax_image_vec
        .iter()
        .map(|v| v * 100.0)
        .collect::<Vec<f32>>();

    let probability_per_image = probability_vec.len() / vec_imgs.len();

    for (i, img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let prob = &probability_vec[start..end];
        // info!("\n\nResults for image: {}\n", img);

        for (i, p) in prob.iter().enumerate() {
            // info!("Probability: {:.4}% Text: {} ", p, vec_seq[i]);
        }
    }

    Ok(())
}

pub fn get_tokenizer(tokenizer: Option<String>) -> anyhow::Result<Tokenizer> {
    let tokenizer = match tokenizer {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.repo(hf_hub::Repo::with_revision(
                "openai/clip-vit-base-patch32".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/15".to_string(),
            ));
            api.get("tokenizer.json")?
        }
        Some(file) => file.into(),
    };

    Tokenizer::from_file(tokenizer).map_err(E::msg)
}

pub fn tokenize_sequences(
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Vec<String>)> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(E::msg("No pad token"))?;

    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "a cycling race".to_string(),
            "a photo of two cats".to_string(),
            "a robot holding a candle".to_string(),
        ],
    };

    let mut tokens = vec![];

    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
        tokens.push(encoding.get_ids().to_vec());
    }

    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);

    // Pad the sequences to have the same length
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }

    let input_ids = Tensor::new(tokens, device)?;

    Ok((input_ids, vec_seq))
}
