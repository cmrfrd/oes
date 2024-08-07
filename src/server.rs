use async_trait::async_trait;
use axum::extract::*;
use axum::routing::get;
use axum::Router;
use axum_extra::extract::CookieJar;
use futures::StreamExt;
use http::Method;
use serde_json::from_str;
use std::fmt::Debug;
use std::io::Cursor;
use tracing::info;

use crate::apis::embeddings::CreateEmbeddingResponse;
use crate::apis::embeddings::Embeddings;
use crate::apis::models::ListModelsResponse;
use crate::apis::models::Models;
use crate::apis::models::RetrieveModelResponse;
use crate::create_model_service_topic;
use crate::dataurl_processor::data_type;
use crate::list_models;
use crate::models::Embedding;
use crate::openai::models;
use crate::pcm_decode_raw;
use crate::types::Nullable;
use crate::Broker;
use crate::DataType;
use crate::EmbeddingMessage;
use crate::Input;
use crate::ModelInfo;
use crate::OesAction;
use crate::OesConfig;
use crate::OesVerb;
use crate::PubSub;
use crate::BAD_REQUEST;
use crate::FOUR_HUNDRED;
use crate::MAX_BATCH_SIZE;
use core::panic;
use data_url::DataUrl;
use rand::Rng;

pub struct OesBaseService {}

impl OesBaseService {
    pub fn new() -> Self {
        info!("Creating OesBaseService");
        Self {}
    }

    pub fn router(self: Self) -> axum::Router {
        Router::new()
            .route("/health", get(|| async { "OK" }))
            .with_state(self)
    }
}

impl Clone for OesBaseService {
    fn clone(&self) -> Self {
        Self {}
    }
}

impl AsRef<OesBaseService> for OesBaseService {
    fn as_ref(&self) -> &OesBaseService {
        self
    }
}

#[derive(Clone)]
pub struct OesOaiService {
    broker_client: Broker<EmbeddingMessage>,
    config: OesConfig,
}

impl OesOaiService {
    pub fn new(broker_client: Broker<EmbeddingMessage>, config: OesConfig) -> Self {
        info!("Creating OesOaiService");
        Self {
            broker_client,
            config,
        }
    }
}

impl AsRef<OesOaiService> for OesOaiService {
    fn as_ref(&self) -> &OesOaiService {
        self
    }
}

struct Guard<T: Debug + Clone + Send + Sync + Unpin + 'static> {
    broker: Broker<T>,
    close_topics: Vec<String>,
}

impl<T: Debug + Clone + Send + Sync + Unpin + 'static> Guard<T> {
    fn new(broker: Broker<T>, close_topics: Vec<String>) -> Self {
        Self {
            broker,
            close_topics,
        }
    }
}

impl<T: Debug + Clone + Send + Sync + Unpin + 'static> Drop for Guard<T> {
    fn drop(&mut self) {
        for topic in self.close_topics.drain(..) {
            self.broker.unsubscribe(topic);
        }
    }
}

#[async_trait]
impl Embeddings for OesOaiService {
    /// Creates an embedding vector representing the input text..
    ///
    /// CreateEmbedding - POST /oai/embeddings
    async fn create_embedding(
        &self,
        _method: Method,
        _host: Host,
        _cookies: CookieJar,
        body: models::CreateEmbeddingRequest,
    ) -> Result<CreateEmbeddingResponse, String> {
        // Parse the model info
        let model_info: ModelInfo = match body.model.parse() {
            Ok(m) => m,
            Err(e) => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present(FOUR_HUNDRED.to_string()),
                        BAD_REQUEST.to_string(),
                        Nullable::Present(format!("Invalid model: {}", e)),
                        "".to_string(),
                    ),
                ));
            }
        };

        // Get the config for said model type
        let model_config = match self
            .config
            .models
            .iter()
            .find(|m| m.model_name == model_info.model)
        {
            Some(m) => m,
            None => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present("404".to_string()),
                        "Not Found".to_string(),
                        Nullable::Present(format!("Model {:?} not found", model_info.model)),
                        "".to_string(),
                    ),
                ));
            }
        };
        let model_encoding_config = match model_config
            .encodings
            .iter()
            .find(|e| e.data_type == model_info.data_type)
        {
            Some(e) => e.clone(),
            None => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present("404".to_string()),
                        "Not Found".to_string(),
                        Nullable::Present(format!("Encoding {:?} not found", model_info.data_type)),
                        "".to_string(),
                    ),
                ));
            }
        };

        // Ensure the input is valid batch size
        let input = {
            let input = serde_json::to_string(&body.input).unwrap();
            if let Ok(single) = from_str::<String>(&input) {
                Ok(vec![single])
            } else if let Ok(multiple) = from_str::<Vec<String>>(&input) {
                if multiple.len() > MAX_BATCH_SIZE {
                    Err(format!("Batch size exceeds maximum of {}", MAX_BATCH_SIZE))
                } else {
                    Ok(multiple)
                }
            } else {
                Err("Invalid input".to_string())
            }
        };
        let input = match input {
            Ok(i) => i,
            Err(e) => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present(FOUR_HUNDRED.to_string()),
                        BAD_REQUEST.to_string(),
                        Nullable::Present(e),
                        "".to_string(),
                    ),
                ));
            }
        };

        let converted_inputs: Result<Vec<Input>, String> = match model_info.data_type {
            DataType::Text => input.iter().try_fold(Vec::new(), |mut acc, i| {
                match DataUrl::process(i) {
                    Ok(data_url) => {
                        let data_type = data_type(&data_url).map_err(|e| e.to_string())?;
                        (data_type == DataType::Text)
                            .then(|| ())
                            .ok_or_else(|| "Invalid input".to_string())?;
                        let (raw_data, _) = data_url.decode_to_vec().map_err(|e| e.to_string())?;
                        acc.push(Input::Text(String::from_utf8_lossy(&raw_data).to_string()));
                    }
                    // If the input is not a data url, assume it's normal text
                    Err(_) => {
                        let owned_i = i.to_owned();
                        acc.push(Input::Text(owned_i));
                    }
                }
                Ok(acc)
            }),
            DataType::Image => input.iter().try_fold(Vec::new(), |mut acc, i| {
                let data_url = DataUrl::process(i).map_err(|e| e.to_string())?;
                let data_type = data_type(&data_url).map_err(|e| e.to_string())?;
                (data_type == DataType::Image)
                    .then(|| ())
                    .ok_or_else(|| "Invalid input".to_string())?;
                let raw_data = data_url.decode_to_vec().map_err(|e| e.to_string())?.0;
                let img = image::load_from_memory(&raw_data).map_err(|e| e.to_string())?;
                acc.push(Input::Image(img));
                Ok(acc)
            }),
            DataType::Audio => input.iter().try_fold(Vec::new(), |mut acc, i| {
                let data_url = DataUrl::process(i).map_err(|e| e.to_string())?;
                let data_type = data_type(&data_url).map_err(|e| e.to_string())?;
                (data_type == DataType::Audio)
                    .then(|| ())
                    .ok_or_else(|| "Invalid input".to_string())?;
                let raw_data = data_url.decode_to_vec().map_err(|e| e.to_string())?.0;
                let media_source = Box::new(Cursor::new(raw_data));
                let (aud, sample_rate) = pcm_decode_raw(media_source).map_err(|e| e.to_string())?;
                acc.push(Input::Audio((aud, sample_rate)));
                Ok(acc)
            }),
        };
        let converted_inputs = match converted_inputs {
            Ok(i) => i,
            Err(e) => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present(FOUR_HUNDRED.to_string()),
                        BAD_REQUEST.to_string(),
                        Nullable::Present(e),
                        "".to_string(),
                    ),
                ))
            }
        };

        // // Ensure that all the inputs (if data urls are needed) are valid
        // let all_valid_inputs = match model_info.data_type {
        //     DataType::Text => true,
        //     DataType::Image => input.iter().all(|i| {
        //         if let Ok(data_url) = DataUrl::process(i) {
        //             if data_url.mime_type().type_ == "image" {
        //                 if vec![String::from("png"), String::from("jpeg")]
        //                     .contains(&data_url.mime_type().subtype)
        //                 {
        //                     true
        //                 } else {
        //                     false
        //                 }
        //             } else {
        //                 false
        //             }
        //         } else {
        //             false
        //         }
        //     }),
        //     DataType::Audio => input.iter().all(|i| {
        //         if let Ok(data_url) = DataUrl::process(i) {
        //             if data_url.mime_type().type_ == "audio" {
        //                 if vec![String::from("wav")].contains(&data_url.mime_type().subtype) {
        //                     true
        //                 } else {
        //                     false
        //                 }
        //             } else {
        //                 false
        //             }
        //         } else {
        //             false
        //         }
        //     }),
        //     _ => {
        //         return Ok(CreateEmbeddingResponse::Status400_BadRequest(
        //             models::Error::new(
        //                 Nullable::Present(FOUR_HUNDRED.to_string()),
        //                 BAD_REQUEST.to_string(),
        //                 Nullable::Present("Unsupported input type".to_string()),
        //                 "".to_string(),
        //             ),
        //         ))
        //     }
        // };
        // if !all_valid_inputs {
        //     return Ok(CreateEmbeddingResponse::Status400_BadRequest(
        //         models::Error::new(
        //             Nullable::Present(FOUR_HUNDRED.to_string()),
        //             BAD_REQUEST.to_string(),
        //             Nullable::Present("Invalid input".to_string()),
        //             "".to_string(),
        //         ),
        //     ));
        // }

        // create a rendezvous topic for pubsub to return
        // and subscribe to it.
        let input_size = input.len();
        let rendezvous_topic = uuid::Uuid::new_v4().to_string();
        let also_rendezvous_topic = rendezvous_topic.clone();
        let _guard = Guard::new(self.broker_client.clone(), vec![rendezvous_topic.clone()]);
        let mut client_send = self.broker_client.clone();
        let mut client_recv = self.broker_client.clone();

        info!("Submitting embeddings request: {}", rendezvous_topic);
        let (_, embeddings) = tokio::join!(
            tokio::spawn(async move {
                let messages =
                    converted_inputs
                        .iter()
                        .cloned()
                        .enumerate()
                        .zip((0..input_size).into_iter().map(|_| {
                            rand::thread_rng().gen::<u32>() % model_encoding_config.replicas
                        }))
                        .map(|((index, input), nonce)| (index, input, nonce))
                        .map(|(index, input, nonce)| {
                            let topic = create_model_service_topic(
                                OesAction::Embed,
                                OesVerb::Request,
                                model_info.model.clone(),
                                model_info.data_type.clone(),
                                nonce,
                            );
                            let req = crate::EmbeddingRequest::new(
                                input.clone(),
                                rendezvous_topic.clone(),
                                index.try_into().unwrap(),
                            );
                            let message = EmbeddingMessage::Request(req);

                            (topic, message)
                        })
                        .collect();
                client_send.publish_many(messages).await.unwrap();
            }),
            tokio::spawn(async move {
                let mut embeddings = Vec::with_capacity(input_size);
                let mut subs_stream = client_recv
                    .subscribe(also_rendezvous_topic.clone())
                    .take(input_size);
                while let Some(entry) = subs_stream.next().await {
                    match entry {
                        EmbeddingMessage::Response(r) => embeddings.push(Embedding::new(
                            r.ordinal.try_into().unwrap(),
                            r.values.clone(),
                            "".to_string(),
                        )),
                        EmbeddingMessage::Error(e) => anyhow::bail!(e),
                        _ => anyhow::bail!("Invalid internal message"),
                    }
                }
                embeddings.sort_by(|a, b| a.index.cmp(&b.index));
                Ok(embeddings)

                // let embeddings: Vec<Embedding> = client_recv
                //     .subscribe(also_rendezvous_topic.clone())
                //     .take(input_size)
                //     .collect::<Vec<_>>()
                //     .await
                //     .into_iter()
                //     .try_fold(Vec::with_capacity(input_size), |mut acc, v| match v {
                //         EmbeddingMessage::Response(r) => {
                //             acc.push(Embedding::new(
                //                 r.ordinal.try_into().unwrap(),
                //                 r.values.clone(),
                //                 "".to_string(),
                //             ));
                //             Ok(acc)
                //         }
                //         EmbeddingMessage::Error(e) => Err(e)?,
                //         // _ => panic!("Invalid response"),
                //         _ => {}
                //     });
                // // embeddings.sort_by(|a, b| a.index.cmp(&b.index));
                // info!("Received embeddings response: {}", also_rendezvous_topic);
                // embeddings
                // Vec::new()
            })
        );

        match embeddings {
            Err(e) => {
                return Ok(CreateEmbeddingResponse::Status500_InternalServerError(
                    models::Error::new(
                        Nullable::Present("500".to_string()),
                        "Internal Server Error".to_string(),
                        Nullable::Present(e.to_string()),
                        "".to_string(),
                    ),
                ));
            }
            Ok(embeddings) => match embeddings {
                Err(e) => {
                    return Ok(CreateEmbeddingResponse::Status500_InternalServerError(
                        models::Error::new(
                            Nullable::Present("500".to_string()),
                            "Internal Server Error".to_string(),
                            Nullable::Present(e.to_string()),
                            "".to_string(),
                        ),
                    ));
                }
                Ok(embeddings) => Ok(CreateEmbeddingResponse::Status200_OK(
                    models::CreateEmbeddingResponse {
                        data: embeddings,
                        model: model_info.model.as_ref().to_string(),
                        object: "".to_string(),
                        usage: models::CreateEmbeddingResponseUsage::new(0, 0),
                    },
                )),
            },
        }
    }
}

#[async_trait]
impl Models for OesOaiService {
    /// Lists the currently available models, and provides basic information about each one such as the owner and availability..
    ///
    /// ListModels - GET /oai/models
    async fn list_models(
        &self,
        _method: Method,
        _host: Host,
        _cookies: CookieJar,
    ) -> Result<ListModelsResponse, String> {
        list_models()
    }

    /// Retrieves a model instance, providing basic information about the model such as the owner and permissioning..
    ///
    /// RetrieveModel - GET /oai/models/{model}
    async fn retrieve_model(
        &self,
        _method: Method,
        _host: Host,
        _cookies: CookieJar,
        path_params: models::RetrieveModelPathParams,
    ) -> Result<RetrieveModelResponse, String> {
        match list_models() {
            Ok(ListModelsResponse::Status200_OK(models)) => {
                let model = models.data.iter().find(|m| m.id == path_params.model);
                match model {
                    Some(m) => Ok(RetrieveModelResponse::Status200_OK(m.clone())),
                    None => Ok(RetrieveModelResponse::Status404_NotFound(
                        models::Error::new(
                            Nullable::Present("404".to_string()),
                            "Not Found".to_string(),
                            Nullable::Present(format!("Model {} not found", path_params.model)),
                            "".to_string(),
                        ),
                    )),
                }
            }
            Ok(ListModelsResponse::Status500_InternalServerError(e)) => {
                Ok(RetrieveModelResponse::Status500_InternalServerError(e))
            }
            Err(e) => Err(e),
            _ => panic!("Invalid response"),
        }
    }
}
