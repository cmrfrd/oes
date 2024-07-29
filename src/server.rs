use async_trait::async_trait;
use axum::extract::*;
use axum::routing::get;
use axum::Router;
use axum_extra::extract::CookieJar;
use futures::StreamExt;
use http::Method;
use serde_json::from_str;
use std::fmt::Debug;
use tracing::info;

use crate::apis::embeddings::CreateEmbeddingResponse;
use crate::apis::embeddings::Embeddings;
use crate::apis::models::ListModelsResponse;
use crate::apis::models::Models;
use crate::apis::models::RetrieveModelResponse;
use crate::create_model_service_topic;
use crate::list_models;
use crate::models::Embedding;
use crate::openai::models;
use crate::types::Nullable;
use crate::Broker;
use crate::DataType;
use crate::EmbeddingMessage;
use crate::OesAction;
use crate::OesConfig;
use crate::OesModel;
use crate::OesVerb;
use crate::PubSub;
use core::panic;
use data_url::DataUrl;
use rand::Rng;

static MAX_BATCH_SIZE: usize = 256;

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
        let (model, encoding) = {
            let parts = body.model.split('/').collect::<Vec<_>>();
            if parts.len() != 3 {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present("400".to_string()),
                        "Bad Request".to_string(),
                        Nullable::Present("Invalid model and encoding".to_string()),
                        "".to_string(),
                    ),
                ));
            }
            let (model, encoding) = (&vec![parts[0], parts[1]].join("/"), &parts[2]);

            match (
                serde_json::from_str::<OesModel>(&serde_json::to_string(model).unwrap()),
                serde_json::from_str::<DataType>(&serde_json::to_string(encoding).unwrap()),
            ) {
                (Ok(m), Ok(e)) => (m, e),
                (Err(_), _) => {
                    return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                        models::Error::new(
                            Nullable::Present("400".to_string()),
                            "Bad Request".to_string(),
                            Nullable::Present(format!("Invalid model: {}", body.model)),
                            "".to_string(),
                        ),
                    ));
                }
                (_, Err(_)) => {
                    return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                        models::Error::new(
                            Nullable::Present("400".to_string()),
                            "Bad Request".to_string(),
                            Nullable::Present(format!("Invalid encoding: {}", body.model)),
                            "".to_string(),
                        ),
                    ));
                }
            }
        };

        let model_config = match self.config.models.iter().find(|m| m.model_name == model) {
            Some(m) => m.clone(),
            None => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present("404".to_string()),
                        "Not Found".to_string(),
                        Nullable::Present(format!("Model {:?} not found", model)),
                        "".to_string(),
                    ),
                ));
            }
        };
        let model_encoding_config = match model_config
            .encodings
            .iter()
            .find(|e| e.data_type == encoding)
        {
            Some(e) => e.clone(),
            None => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present("404".to_string()),
                        "Not Found".to_string(),
                        Nullable::Present(format!("Encoding {:?} not found", encoding)),
                        "".to_string(),
                    ),
                ));
            }
        };

        // Ensure the input is valid
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
                        Nullable::Present("400".to_string()),
                        "Bad Request".to_string(),
                        Nullable::Present(e),
                        "".to_string(),
                    ),
                ));
            }
        };

        // Ensure that all the inputs (if data urls are needed) are valid
        let all_valid_inputs = match encoding {
            DataType::Text => true,
            DataType::Image => input.iter().all(|i| {
                if let Ok(data_url) = DataUrl::process(i) {
                    if data_url.mime_type().type_ == "image" {
                        if vec![String::from("png"), String::from("jpeg")]
                            .contains(&data_url.mime_type().subtype)
                        {
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            }),
            _ => {
                return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                    models::Error::new(
                        Nullable::Present("400".to_string()),
                        "Bad Request".to_string(),
                        Nullable::Present("Unsupported input type".to_string()),
                        "".to_string(),
                    ),
                ))
            }
        };
        if !all_valid_inputs {
            return Ok(CreateEmbeddingResponse::Status400_BadRequest(
                models::Error::new(
                    Nullable::Present("400".to_string()),
                    "Bad Request".to_string(),
                    Nullable::Present("Invalid input".to_string()),
                    "".to_string(),
                ),
            ));
        }

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
                let messages = input
                    .iter()
                    .map(|i| {
                        let is_data_url = DataUrl::process(&i).is_ok();
                        let nonce: u32 =
                            rand::thread_rng().gen::<u32>() % model_encoding_config.replicas;
                        (
                            create_model_service_topic(
                                OesAction::Embed,
                                OesVerb::Request,
                                model.clone(),
                                if is_data_url {
                                    DataType::Image
                                } else {
                                    DataType::Text
                                },
                                nonce,
                            ),
                            EmbeddingMessage::Request(crate::EmbeddingRequest {
                                model: "clip".to_string(),
                                input: i.clone(),
                                rendezvous_topic: rendezvous_topic.clone(),
                            }),
                        )
                    })
                    .collect::<Vec<_>>();
                client_send.publish_many(messages).await.unwrap();
            }),
            tokio::spawn(async move {
                let mut embeddings = Vec::with_capacity(input_size);
                client_recv
                    .subscribe(also_rendezvous_topic.clone())
                    .take(input_size)
                    .collect::<Vec<_>>()
                    .await
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, v)| match v {
                        EmbeddingMessage::Response(r) => {
                            embeddings.push(Embedding::new(
                                i.try_into().unwrap(),
                                r.values.clone(),
                                "".to_string(),
                            ));
                        }
                        _ => {
                            panic!("Invalid response");
                        }
                    });
                embeddings
            })
        );

        return Ok(CreateEmbeddingResponse::Status200_OK(
            models::CreateEmbeddingResponse {
                data: embeddings.unwrap(),
                model: "clip".to_string(),
                object: "".to_string(),
                usage: models::CreateEmbeddingResponseUsage::new(0, 0),
            },
        ));
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
