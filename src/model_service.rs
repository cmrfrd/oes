use candle_core::quantized::gguf_file;
use candle_transformers::models::clip::{self, ClipConfig};
use candle_transformers::models::qwen2::{self, Config, Model};
use data_url::DataUrl;
use hf_hub::api::sync::Api;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};
use tracing::{error, info, warn};

use crate::{
    create_model_service_topic, Broker, DataType, EmbeddingMessage, OesAction, OesConfig, OesModel,
    OesVerb, PubSub, MAX_BATCH_SIZE_CLIP_TEXT,
};

use anyhow::Error as E;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

pub struct OesModelService {
    broker: Broker<EmbeddingMessage>,
    config: OesConfig,
}

impl OesModelService {
    pub fn new(broker: Broker<EmbeddingMessage>, config: OesConfig) -> Self {
        Self { broker, config }
    }

    pub fn run(self) {
        if self.config.models.is_empty() {
            warn!("OesModelService has no models to run");
            return;
        }
        info!("Starting OesModelService");
        for model_config in self.config.models.iter() {
            for encodings_config in model_config.encodings.iter() {
                match (&model_config.model_name, &encodings_config.data_type) {
                    (OesModel::OpenAIClip, DataType::Text) => {
                        for replica_id in 0..encodings_config.replicas {
                            let (tokenizer, model, _, device) = init_clip();
                            let mut local_broker = self.broker.clone();
                            let receive_topic = create_model_service_topic(
                                OesAction::Embed,
                                OesVerb::Request,
                                OesModel::OpenAIClip,
                                DataType::Text,
                                replica_id,
                            );
                            info!(
                                "CLIP Model Text Service replica {} started listening on {}",
                                replica_id, receive_topic
                            );
                            tokio::spawn(async move {
                                loop {
                                    // Collect messages from broker, filter out non-request messages,
                                    // and continue if no messages are received
                                    let received_messages = local_broker
                                        .try_recv_many(
                                            receive_topic.clone(),
                                            MAX_BATCH_SIZE_CLIP_TEXT,
                                        )
                                        .await
                                        .into_iter()
                                        .filter_map(|m| match m {
                                            EmbeddingMessage::Request(r) => Some(r),
                                            _ => {
                                                error!("Received non-request message: {:?}", m);
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    if received_messages.is_empty() {
                                        tokio::task::yield_now().await;
                                        continue;
                                    }

                                    // tokenization and processing of the received messages
                                    let (input_ids, _) = tokenize_sequences(
                                        received_messages
                                            .iter()
                                            .map(|r| r.input.clone())
                                            .collect::<Vec<String>>(),
                                        &tokenizer,
                                        &device,
                                    )
                                    .unwrap();
                                    tokio::task::yield_now().await;

                                    // Get the text features
                                    let text_features = model
                                        .get_text_features(&input_ids)
                                        .unwrap()
                                        .to_vec2::<f32>()
                                        .unwrap();
                                    tokio::task::yield_now().await;

                                    // Send the responses back to the broker
                                    let mut send_broker = local_broker.clone();
                                    tokio::spawn(async move {
                                        let responses = received_messages
                                            .iter()
                                            .map(|r| r.rendezvous_topic.to_string())
                                            .zip(text_features.iter().map(|f| {
                                                EmbeddingMessage::Response(
                                                    crate::EmbeddingResponse {
                                                        values: f
                                                            .iter()
                                                            .map(|v| *v as f64)
                                                            .collect(),
                                                    },
                                                )
                                            }))
                                            .collect::<Vec<_>>();
                                        send_broker.publish_many(responses).await.unwrap();
                                    });
                                }
                            });
                        }
                    }
                    (OesModel::OpenAIClip, DataType::Image) => {
                        for replica_id in 0..encodings_config.replicas {
                            let (_, model, config, device) = init_clip();
                            let mut local_broker = self.broker.clone();
                            let receive_topic = create_model_service_topic(
                                OesAction::Embed,
                                OesVerb::Request,
                                OesModel::OpenAIClip,
                                DataType::Image,
                                replica_id,
                            );
                            info!(
                                "CLIP Model Image Service replica {} started listening on {}",
                                replica_id, receive_topic
                            );
                            tokio::spawn(async move {
                                loop {
                                    // Collect messages from broker, filter out non-request messages,
                                    // and continue if no messages are received
                                    let received_messages = local_broker
                                        .try_recv_many(
                                            receive_topic.clone(),
                                            MAX_BATCH_SIZE_CLIP_TEXT,
                                        )
                                        .await
                                        .into_iter()
                                        .filter_map(|m| match m {
                                            EmbeddingMessage::Request(r) => Some(r),
                                            _ => {
                                                error!("Received non-request message: {:?}", m);
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    if received_messages.is_empty() {
                                        tokio::task::yield_now().await;
                                        continue;
                                    }

                                    // Get the image features
                                    let images = received_messages
                                        .iter()
                                        .map(|r| r.input.clone())
                                        .map(|i| {
                                            DataUrl::process(&i.clone())
                                                .unwrap()
                                                .decode_to_vec()
                                                .unwrap()
                                                .0
                                        })
                                        .map(|data| image::load_from_memory(&data).unwrap())
                                        .map(|img| {
                                            let (height, width) =
                                                (config.image_size, config.image_size);
                                            let img = img.resize_to_fill(
                                                width as u32,
                                                height as u32,
                                                image::imageops::FilterType::Triangle,
                                            );
                                            let img = img.to_rgb8();
                                            let img = img.into_raw();
                                            let img = Tensor::from_vec(
                                                img,
                                                (height, width, 3),
                                                &Device::Cpu,
                                            )
                                            .unwrap()
                                            .permute((2, 0, 1))
                                            .unwrap()
                                            .to_dtype(DType::F32)
                                            .unwrap()
                                            .affine(2. / 255., -1.)
                                            .unwrap();
                                            // .unsqueeze(0)?;
                                            img
                                        })
                                        .collect::<Vec<_>>();
                                    tokio::task::yield_now().await;

                                    // Get the image features
                                    let image_features = model
                                        .get_image_features(
                                            &Tensor::stack(&images, 0)
                                                .unwrap()
                                                .to_device(&device)
                                                .unwrap(),
                                        )
                                        .unwrap()
                                        .to_vec2::<f32>()
                                        .unwrap();
                                    tokio::task::yield_now().await;

                                    // Send the responses back to the broker
                                    let mut send_broker = local_broker.clone();
                                    tokio::spawn(async move {
                                        let responses = received_messages
                                            .iter()
                                            .map(|r| r.rendezvous_topic.to_string())
                                            .zip(image_features.iter().map(|f| {
                                                EmbeddingMessage::Response(
                                                    crate::EmbeddingResponse {
                                                        values: f
                                                            .iter()
                                                            .map(|v| *v as f64)
                                                            .collect(),
                                                    },
                                                )
                                            }))
                                            .collect::<Vec<_>>();
                                        send_broker.publish_many(responses).await.unwrap();
                                    });
                                }
                            });
                        }
                    }
                    (OesModel::GteQwen, DataType::Text) => {
                        for replica_id in 0..encodings_config.replicas {
                            let (tokenizer, mut model, device) = init_gtwqwen();
                            let mut local_broker = self.broker.clone();
                            let receive_topic = create_model_service_topic(
                                OesAction::Embed,
                                OesVerb::Request,
                                OesModel::GteQwen,
                                DataType::Text,
                                replica_id,
                            );
                            info!(
                                "{} Model Text Service replica {} started listening on {}",
                                serde_json::to_string(&OesModel::GteQwen)
                                    .unwrap()
                                    .as_str()
                                    .trim_matches('"')
                                    .to_string(),
                                replica_id,
                                receive_topic
                            );
                            tokio::spawn(async move {
                                loop {
                                    // Collect messages from broker, filter out non-request messages,
                                    // and continue if no messages are received
                                    let received_messages = local_broker
                                        .try_recv_many(
                                            receive_topic.clone(),
                                            MAX_BATCH_SIZE_CLIP_TEXT,
                                        )
                                        .await
                                        .into_iter()
                                        .filter_map(|m| match m {
                                            EmbeddingMessage::Request(r) => Some(r),
                                            _ => {
                                                error!("Received non-request message: {:?}", m);
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    if received_messages.is_empty() {
                                        tokio::task::yield_now().await;
                                        continue;
                                    }
                                    info!("Received {} messages", received_messages.len());

                                    let encoded = tokenizer
                                        .encode_batch(
                                            received_messages
                                                .iter()
                                                .map(|r| r.input.clone())
                                                .collect::<Vec<String>>(),
                                            true,
                                        )
                                        .map_err(anyhow::Error::msg)
                                        .unwrap();
                                    let tokens: Vec<&[u32]> =
                                        encoded.iter().map(|x| x.get_ids()).collect();
                                    let tokens = Tensor::new(tokens, &device).unwrap();
                                    let mask: Vec<&[u32]> =
                                        encoded.iter().map(|x| x.get_attention_mask()).collect();
                                    let mask = Tensor::new(mask, &device).unwrap();
                                    let logits = model.forward(&tokens, 0, Some(&mask)).unwrap();
                                    let (_, seq_len, _) = logits.dims3().unwrap();
                                    let embd = logits
                                        .narrow(1, seq_len - 1, 1)
                                        .unwrap()
                                        .squeeze(1)
                                        .unwrap()
                                        .to_dtype(DType::F32)
                                        .unwrap();
                                    let norm = embd
                                        .broadcast_div(
                                            &embd
                                                .sqr()
                                                .unwrap()
                                                .sum_keepdim(1)
                                                .unwrap()
                                                .sqrt()
                                                .unwrap(),
                                        )
                                        .unwrap();
                                    tokio::task::yield_now().await;

                                    // Get the text features
                                    let text_features = norm.to_vec2::<f32>().unwrap();
                                    tokio::task::yield_now().await;

                                    // Send the responses back to the broker
                                    let mut send_broker = local_broker.clone();
                                    tokio::spawn(async move {
                                        let responses = received_messages
                                            .iter()
                                            .map(|r| r.rendezvous_topic.to_string())
                                            .zip(text_features.iter().map(|f| {
                                                EmbeddingMessage::Response(
                                                    crate::EmbeddingResponse {
                                                        values: f
                                                            .iter()
                                                            .map(|v| *v as f64)
                                                            .collect(),
                                                    },
                                                )
                                            }))
                                            .collect::<Vec<_>>();
                                        send_broker.publish_many(responses).await.unwrap();
                                    });
                                }
                            });
                        }
                    }
                    _ => panic!(
                        "Model '{:?}' + input type '{:?}' not supported",
                        model_config.model_name, encodings_config.data_type
                    ),
                }
            }
        }
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle_core::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle_core::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors_files)
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn tokenize_sequences(
    sequences: Vec<String>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Vec<String>)> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(E::msg("No pad token"))?;

    let mut tokens = vec![];

    for seq in sequences.clone().into_iter() {
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

    Ok((input_ids, sequences))
}

pub fn init_clip() -> (Tokenizer, clip::ClipModel, ClipConfig, Device) {
    let api = hf_hub::api::sync::Api::new().unwrap();
    let api = api.repo(hf_hub::Repo::with_revision(
        "openai/clip-vit-base-patch32".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/15".to_string(),
    ));
    let model_file = api.get("model.safetensors").unwrap();
    let tokenizer = Tokenizer::from_file(api.get("tokenizer.json").unwrap())
        .map_err(E::msg)
        .unwrap();
    let config = clip::ClipConfig::vit_base_patch32();
    let device = device(false).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device).unwrap()
    };
    let model = clip::ClipModel::new(vb, &config).unwrap();
    (tokenizer, model, config, device)
}

pub fn init_gtwqwen() -> (Tokenizer, qwen2::Model, Device) {
    let device = device(false).unwrap();
    let api = hf_hub::api::sync::Api::new().unwrap();
    let repo = api.repo(hf_hub::Repo::new(
        "Alibaba-NLP/gte-Qwen1.5-7B-instruct".to_string(),
        hf_hub::RepoType::Model,
    ));
    info!("info {:?}", repo.info().unwrap());

    let config = repo.get("config.json").unwrap();
    let tokenizer = repo.get("tokenizer.json").unwrap();
    let weights = hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap();

    const EOS_TOKEN: &str = "<|endoftext|>";
    const EOS_TOKEN_ID: u32 = 151643;
    let padding = PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Left,
        pad_to_multiple_of: None,
        pad_id: EOS_TOKEN_ID,
        pad_type_id: 0,
        pad_token: String::from(EOS_TOKEN),
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg).unwrap();
    tokenizer.with_padding(Some(padding));
    let config: Config = serde_json::from_slice(&std::fs::read(config).unwrap()).unwrap();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &device).unwrap() };
    let model = Model::new(&config, vb).unwrap();

    (tokenizer, model, device)
}
