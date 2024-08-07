use std::path::{Path, PathBuf};

use candle_transformers::models::clip::{self, ClipConfig};
use candle_transformers::models::qwen2::{Config as QwenConfig, Model as QwenModel};
use candle_transformers::models::whisper::{self as m, audio, Config};
use itertools::Itertools;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};
use tracing::{error, info, warn};

use crate::{
    create_model_service_topic, Broker, DataType, EmbeddingMessage, OesAction, OesConfig, OesModel,
    OesVerb, PubSub, MAX_BATCH_SIZE,
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
                                        .try_recv_many(receive_topic.clone(), MAX_BATCH_SIZE)
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

                                    // ensure the request inputs are valid, for invalid requests send an err
                                    // to their rendezvous topics if they exist
                                    let mut send_err_broker = local_broker.clone();
                                    let (valid_received_messages, invalid_received_messages): (
                                        Vec<_>,
                                        Vec<_>,
                                    ) = received_messages.into_iter().partition(
                                        |received_message| match received_message.payload {
                                            crate::Input::Text(_) => true,
                                            _ => {
                                                error!(
                                                    "Received non-text input: {:?}",
                                                    received_message
                                                );
                                                false
                                            }
                                        },
                                    );
                                    let valid_received_messages_headers = valid_received_messages
                                        .iter()
                                        .map(|r| r.header.clone())
                                        .collect::<Vec<_>>();
                                    let inputs = valid_received_messages
                                        .into_iter()
                                        .map(|r| match r.payload {
                                            crate::Input::Text(t) => t,
                                            _ => unreachable!(),
                                        })
                                        .collect::<Vec<_>>();
                                    tokio::spawn(async move {
                                        let responses = invalid_received_messages
                                            .into_iter()
                                            .map(|r| r.header.rendezvous_topic.to_string())
                                            .unique()
                                            .filter(|r| send_err_broker.has_topic(r))
                                            .map(|r| {
                                                (
                                                    r.to_string(),
                                                    EmbeddingMessage::Error(
                                                        "Invalid input".to_string(),
                                                    ),
                                                )
                                            })
                                            .collect::<Vec<_>>();
                                        send_err_broker.publish_many(responses).await.unwrap();
                                    });

                                    // Process embeddings
                                    let (input_ids, _) =
                                        tokenize_sequences(inputs, &tokenizer, &device).unwrap();
                                    let text_embeddings = model
                                        .get_text_features(&input_ids)
                                        .unwrap()
                                        .to_vec2::<f32>()
                                        .unwrap();
                                    tokio::task::yield_now().await;

                                    // Send the responses back to the broker
                                    let mut send_broker = local_broker.clone();
                                    tokio::spawn(async move {
                                        let responses = valid_received_messages_headers
                                            .iter()
                                            .zip(text_embeddings.iter())
                                            .map(|(header, embedding)| {
                                                (
                                                    header.rendezvous_topic.to_string(),
                                                    EmbeddingMessage::Response(
                                                        crate::EmbeddingResponse {
                                                            values: embedding
                                                                .iter()
                                                                .map(|v| *v as f64)
                                                                .collect(),
                                                            ordinal: header.ordinal,
                                                        },
                                                    ),
                                                )
                                            })
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
                                        .try_recv_many(receive_topic.clone(), MAX_BATCH_SIZE)
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

                                    // ensure the request inputs are valid, for invalid requests send an err
                                    // to their rendezvous topics if they exist
                                    let mut send_err_broker = local_broker.clone();
                                    let (valid_received_messages, invalid_received_messages): (
                                        Vec<_>,
                                        Vec<_>,
                                    ) = received_messages.into_iter().partition(
                                        |received_message| match received_message.payload {
                                            crate::Input::Image(_) => true,
                                            _ => {
                                                error!(
                                                    "Received non-image input: {:?}",
                                                    received_message
                                                );
                                                false
                                            }
                                        },
                                    );
                                    let valid_received_messages_headers = valid_received_messages
                                        .iter()
                                        .map(|r| r.header.clone())
                                        .collect::<Vec<_>>();
                                    let inputs = valid_received_messages
                                        .into_iter()
                                        .map(|r| match r.payload {
                                            crate::Input::Image(t) => t,
                                            _ => unreachable!(),
                                        })
                                        .collect::<Vec<_>>();
                                    tokio::spawn(async move {
                                        let responses = invalid_received_messages
                                            .into_iter()
                                            .map(|r| r.header.rendezvous_topic.to_string())
                                            .unique()
                                            .filter(|r| send_err_broker.has_topic(r))
                                            .map(|r| {
                                                (
                                                    r.to_string(),
                                                    EmbeddingMessage::Error(
                                                        "Invalid input".to_string(),
                                                    ),
                                                )
                                            })
                                            .collect::<Vec<_>>();
                                        send_err_broker.publish_many(responses).await.unwrap();
                                    });

                                    // Get the image features
                                    let images = inputs
                                        .iter()
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
                                    let image_embeddings = model
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
                                        let responses = valid_received_messages_headers
                                            .iter()
                                            .zip(image_embeddings.iter())
                                            .map(|(header, embedding)| {
                                                (
                                                    header.rendezvous_topic.to_string(),
                                                    EmbeddingMessage::Response(
                                                        crate::EmbeddingResponse {
                                                            values: embedding
                                                                .iter()
                                                                .map(|v| *v as f64)
                                                                .collect(),
                                                            ordinal: header.ordinal,
                                                        },
                                                    ),
                                                )
                                            })
                                            .collect::<Vec<_>>();
                                        send_broker.publish_many(responses).await.unwrap();
                                    });
                                }
                            });
                        }
                    }
                    (OesModel::GteQwen, DataType::Text) => {
                        for replica_id in 0..encodings_config.replicas {
                            let (tokenizer, mut model, device) = init_gteqwen();
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
                                        .try_recv_many(receive_topic.clone(), MAX_BATCH_SIZE)
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
                                    info!("Received {:?} messages", received_messages.len());

                                    // ensure the request inputs are valid, for invalid requests send an err
                                    // to their rendezvous topics if they exist
                                    let mut send_err_broker = local_broker.clone();
                                    let (valid_received_messages, invalid_received_messages): (
                                        Vec<_>,
                                        Vec<_>,
                                    ) = received_messages.into_iter().partition(
                                        |received_message| match received_message.payload {
                                            crate::Input::Text(_) => true,
                                            _ => {
                                                error!(
                                                    "Received non-text input: {:?}",
                                                    received_message
                                                );
                                                false
                                            }
                                        },
                                    );
                                    let valid_received_messages_headers = valid_received_messages
                                        .iter()
                                        .map(|r| r.header.clone())
                                        .collect::<Vec<_>>();
                                    let inputs = valid_received_messages
                                        .into_iter()
                                        .map(|r| match r.payload {
                                            crate::Input::Text(t) => t,
                                            _ => unreachable!(),
                                        })
                                        .collect::<Vec<_>>();
                                    tokio::spawn(async move {
                                        let responses = invalid_received_messages
                                            .into_iter()
                                            .map(|r| r.header.rendezvous_topic.to_string())
                                            .unique()
                                            .filter(|r| send_err_broker.has_topic(r))
                                            .map(|r| {
                                                (
                                                    r.to_string(),
                                                    EmbeddingMessage::Error(
                                                        "Invalid input".to_string(),
                                                    ),
                                                )
                                            })
                                            .collect::<Vec<_>>();
                                        send_err_broker.publish_many(responses).await.unwrap();
                                    });

                                    // Process embeddings
                                    let inputs = inputs
                                        .into_iter()
                                        .map(|m| {
                                            // Very strange hack, but is needed for good results
                                            if m.ends_with("<|endoftext|>") {
                                                m
                                            } else {
                                                format!("{}<|endoftext|>", m)
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    let encoded = tokenizer
                                        .encode_batch(inputs, true)
                                        .map_err(E::msg)
                                        .unwrap();
                                    let tokens: Vec<&[u32]> =
                                        encoded.iter().map(|x| x.get_ids()).collect();
                                    let tokens = Tensor::new(tokens, &device).unwrap();
                                    let mask: Vec<&[u32]> =
                                        encoded.iter().map(|x| x.get_attention_mask()).collect();
                                    let mask = Tensor::new(mask, &device).unwrap();
                                    let logits = model.forward(&tokens, 0, Some(&mask)).unwrap();
                                    model.clear_kv_cache(); // this is important to avoid memory leak and crashing
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
                                    let text_embeddings = norm.to_vec2::<f32>().unwrap();
                                    tokio::task::yield_now().await;

                                    // Send the responses back to the broker
                                    let mut send_broker = local_broker.clone();
                                    tokio::spawn(async move {
                                        let responses = valid_received_messages_headers
                                            .iter()
                                            .zip(text_embeddings.iter())
                                            .map(|(header, embedding)| {
                                                (
                                                    header.rendezvous_topic.to_string(),
                                                    EmbeddingMessage::Response(
                                                        crate::EmbeddingResponse {
                                                            values: embedding
                                                                .iter()
                                                                .map(|v| *v as f64)
                                                                .collect(),
                                                            ordinal: header.ordinal,
                                                        },
                                                    ),
                                                )
                                            })
                                            .collect::<Vec<_>>();
                                        send_broker.publish_many(responses).await.unwrap();
                                    });
                                }
                            });
                        }
                    }
                    (OesModel::Whisper, DataType::Audio) => {
                        for replica_id in 0..encodings_config.replicas {
                            let path = assets_file_path("melfilters128.bytes");
                            let mel_bytes = std::fs::read(path).unwrap();
                            let mel_bytes_slice = mel_bytes.as_slice();

                            let mut mel_filters = vec![0f32; mel_bytes_slice.len() / 4];
                            <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
                                mel_bytes_slice,
                                &mut mel_filters,
                            );

                            let (config, _tokenizer, mut model, device) = init_whisper();
                            let mut local_broker = self.broker.clone();
                            let receive_topic = create_model_service_topic(
                                OesAction::Embed,
                                OesVerb::Request,
                                OesModel::Whisper,
                                DataType::Audio,
                                replica_id,
                            );
                            info!(
                                "{} Model Audio Service replica {} started listening on {}",
                                serde_json::to_string(&OesModel::Whisper)
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
                                        .try_recv_many(receive_topic.clone(), MAX_BATCH_SIZE)
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
                                    info!("Received {:?} messages", received_messages.len());

                                    // ensure the request inputs are valid, for invalid requests send an err
                                    // to their rendezvous topics if they exist
                                    let mut send_err_broker = local_broker.clone();
                                    let (valid_received_messages, invalid_received_messages): (
                                        Vec<_>,
                                        Vec<_>,
                                    ) = received_messages.into_iter().partition(
                                        |received_message| match received_message.payload {
                                            crate::Input::Audio(_) => true,
                                            _ => {
                                                error!(
                                                    "Received non-audio input: {:?}",
                                                    received_message
                                                );
                                                false
                                            }
                                        },
                                    );
                                    let valid_received_messages_headers = valid_received_messages
                                        .iter()
                                        .map(|r| r.header.clone())
                                        .collect::<Vec<_>>();
                                    let inputs = valid_received_messages
                                        .into_iter()
                                        .map(|r| match r.payload {
                                            crate::Input::Audio(t) => t,
                                            _ => unreachable!(),
                                        })
                                        .collect::<Vec<_>>();
                                    tokio::spawn(async move {
                                        let responses = invalid_received_messages
                                            .into_iter()
                                            .map(|r| r.header.rendezvous_topic.to_string())
                                            .unique()
                                            .filter(|r| send_err_broker.has_topic(r))
                                            .map(|r| {
                                                (
                                                    r.to_string(),
                                                    EmbeddingMessage::Error(
                                                        "Invalid input".to_string(),
                                                    ),
                                                )
                                            })
                                            .collect::<Vec<_>>();
                                        send_err_broker.publish_many(responses).await.unwrap();
                                    });

                                    // Process embeddings
                                    let embedding_tensors = inputs
                                        .into_iter()
                                        .map(|(pcm_data, _)| {
                                            let mel =
                                                audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
                                            let mel_len = mel.len();
                                            let mel = Tensor::from_vec(
                                                mel,
                                                (
                                                    1,
                                                    config.num_mel_bins,
                                                    mel_len / config.num_mel_bins,
                                                ),
                                                &device,
                                            )
                                            .unwrap();

                                            let mut seek = 0;
                                            let mut embeddings = vec![];
                                            let (_, _, content_frames) = mel.dims3().unwrap();
                                            while seek < content_frames {
                                                let _time_offset = (seek * m::HOP_LENGTH) as f64
                                                    / m::SAMPLE_RATE as f64;
                                                let segment_size =
                                                    usize::min(content_frames - seek, m::N_FRAMES);
                                                let mel_segment =
                                                    mel.narrow(2, seek, segment_size).unwrap();
                                                let _segment_duration =
                                                    (segment_size * m::HOP_LENGTH) as f64
                                                        / m::SAMPLE_RATE as f64;
                                                let embedding = model
                                                    .encoder
                                                    .forward(&mel_segment, true)
                                                    .unwrap();
                                                info!(
                                                    "Embedding segment shape: {:?}",
                                                    embedding.dims()
                                                );
                                                model.reset_kv_cache();
                                                seek += segment_size;
                                                embeddings.push(embedding);
                                            }

                                            let all_embedding_tensors = Tensor::cat(&embeddings, 1)
                                                .unwrap()
                                                .to_device(&device)
                                                .unwrap();

                                            let (_n_audio, sequences, n_dim) =
                                                all_embedding_tensors.dims3().unwrap();
                                            let mean_embeddings = all_embedding_tensors
                                                .sum(1)
                                                .unwrap()
                                                .div(
                                                    &Tensor::new(vec![sequences as f32], &device)
                                                        .unwrap()
                                                        .broadcast_as(vec![1, n_dim])
                                                        .unwrap(),
                                                )
                                                .unwrap();
                                            mean_embeddings
                                        })
                                        .collect::<Vec<_>>();
                                    let embedding_tensors = Tensor::cat(&embedding_tensors, 0)
                                        .unwrap()
                                        .to_device(&device)
                                        .unwrap();
                                    let audio_embeddings =
                                        embedding_tensors.to_vec2::<f32>().unwrap();
                                    tokio::task::yield_now().await;

                                    // Send the responses back to the broker
                                    let mut send_broker = local_broker.clone();
                                    tokio::spawn(async move {
                                        let responses = valid_received_messages_headers
                                            .iter()
                                            .zip(audio_embeddings.iter())
                                            .map(|(header, embedding)| {
                                                (
                                                    header.rendezvous_topic.to_string(),
                                                    EmbeddingMessage::Response(
                                                        crate::EmbeddingResponse {
                                                            values: embedding
                                                                .iter()
                                                                .map(|v| *v as f64)
                                                                .collect(),
                                                            ordinal: header.ordinal,
                                                        },
                                                    ),
                                                )
                                            })
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

pub fn init_gteqwen() -> (Tokenizer, QwenModel, Device) {
    let device = device(false).unwrap();
    let api = hf_hub::api::sync::Api::new().unwrap();
    let repo = api.repo(hf_hub::Repo::new(
        "Alibaba-NLP/gte-Qwen1.5-7B-instruct".to_string(),
        hf_hub::RepoType::Model,
    ));

    let config = repo.get("config.json").unwrap();
    let tokenizer = repo.get("tokenizer.json").unwrap();
    let model_file = hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap();

    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg).unwrap();
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
    tokenizer.with_padding(Some(padding));

    let config: QwenConfig = serde_json::from_slice(&std::fs::read(config).unwrap()).unwrap();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&model_file, DType::F32, &device).unwrap() };
    let model = QwenModel::new(&config, vb).unwrap();
    (tokenizer, model, device)
}

pub fn init_whisper() -> (Config, Tokenizer, m::model::Whisper, Device) {
    let device = device(false).unwrap();
    let api = hf_hub::api::sync::Api::new().unwrap();
    let repo = api.repo(hf_hub::Repo::with_revision(
        "openai/whisper-large-v2".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/57".to_string(),
    ));

    let config = repo.get("config.json").unwrap();
    let tokenizer = repo.get("tokenizer.json").unwrap();
    let model = repo.get("model.safetensors").unwrap();

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg).unwrap();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], m::DTYPE, &device).unwrap() };
    let model = m::model::Whisper::load(&vb, config.clone()).unwrap();
    (config, tokenizer, model, device)
}

pub fn assets_file_path(name: &str) -> PathBuf {
    let base_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let assets_path = base_path.join("assets");
    assets_path.join(name)
}
