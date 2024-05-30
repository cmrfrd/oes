use candle_transformers::models::clip::{self, ClipConfig};
use data_url::DataUrl;
use tokenizers::Tokenizer;
use tracing::{error, info};

use crate::{
    create_model_service_topic, Broker, EmbeddingMessage, InputType, OesAction, OesConfig,
    OesModel, OesVerb, PubSub, MAX_BATCH_SIZE_CLIP_TEXT,
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
        info!("Creating OesModelService");
        Self { broker, config }
    }

    pub fn run(self) {
        info!("Starting OesModelService");

        for model_config in self.config.models.iter() {
            match (
                &model_config.model_name,
                InputType::from(model_config.input_types.clone()),
            ) {
                (OesModel::OpenAIClip, InputType::Text) => {
                    for replica_id in 0..model_config.replicas {
                        let (tokenizer, model, _, device) = init_clip();
                        let mut local_broker = self.broker.clone();
                        let receive_topic = create_model_service_topic(
                            OesAction::Embed,
                            OesVerb::Request,
                            OesModel::OpenAIClip,
                            InputType::Text,
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
                                    .try_recv_many(receive_topic.clone(), MAX_BATCH_SIZE_CLIP_TEXT)
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
                                            EmbeddingMessage::Response(crate::EmbeddingResponse {
                                                values: f.iter().map(|v| *v as f64).collect(),
                                            })
                                        }))
                                        .collect::<Vec<_>>();
                                    send_broker.publish_many(responses).await.unwrap();
                                });
                            }
                        });
                    }
                }
                (OesModel::OpenAIClip, InputType::Image) => {
                    for replica_id in 0..model_config.replicas {
                        let (_, model, config, device) = init_clip();
                        let mut local_broker = self.broker.clone();
                        let receive_topic = create_model_service_topic(
                            OesAction::Embed,
                            OesVerb::Request,
                            OesModel::OpenAIClip,
                            InputType::Image,
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
                                    .try_recv_many(receive_topic.clone(), MAX_BATCH_SIZE_CLIP_TEXT)
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
                                        let img =
                                            Tensor::from_vec(img, (height, width, 3), &Device::Cpu)
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
                                            EmbeddingMessage::Response(crate::EmbeddingResponse {
                                                values: f.iter().map(|v| *v as f64).collect(),
                                            })
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
                    model_config.model_name, model_config.input_types
                ),
            }
        }
    }
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
