#![feature(const_trait_impl)]
#![feature(effects)]

pub mod broker;
pub mod config;
pub mod messages;
pub mod model_service;
pub mod openai;
pub mod server;

use anyhow::Error;
use openai::apis::models::ListModelsResponse;
use openai::types::Nullable;

pub use crate::broker::*;
pub use crate::config::*;
pub use crate::messages::*;
pub use crate::model_service::*;
pub use crate::openai::*;
pub use crate::server::*;

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

// Update this map with more model support
#[inline]
pub fn model_compatability_map() -> HashMap<OesModel, Vec<InputType>> {
    let mut map = HashMap::new();
    map.insert(
        OesModel::OpenAIClip,
        vec![InputType::Text, InputType::Image],
    );
    map
}

#[inline]
pub fn create_model_ids() -> Result<Vec<String>, anyhow::Error> {
    let map = model_compatability_map();
    let mut ids = Vec::new();
    for (model, _) in map.iter() {
        let model = serde_json::Value::from_str(&serde_json::to_string(&model)?)?
            .as_str()
            .ok_or(Error::msg("Invalid model id"))?
            .to_string();
        ids.push(model);
    }
    Ok(ids)
}

pub fn list_models() -> Result<ListModelsResponse, String> {
    match create_model_ids() {
        Ok(model_ids) => Ok(ListModelsResponse::Status200_OK(
            models::ListModelsResponse::new(
                "list".to_owned(),
                model_ids
                    .iter()
                    .map(|m| models::Model::new(m.to_owned(), 0, "".to_owned(), OWNER.to_string()))
                    .collect(),
            ),
        )),
        Err(e) => Ok(ListModelsResponse::Status500_InternalServerError(
            models::Error::new(
                Nullable::Present("500".to_string()),
                "Internal Server Error".to_string(),
                Nullable::Present(e.to_string()),
                "".to_string(),
            ),
        )),
    }
}

pub fn create_model_service_topic(
    action: OesAction,
    verb: OesVerb,
    model: OesModel,
    input_type: InputType,
    replica_id: u32,
) -> String {
    Path::new(
        &serde_json::to_string(&action)
            .unwrap()
            .as_str()
            .trim_matches('"')
            .to_string(),
    )
    .join(
        serde_json::to_string(&verb)
            .unwrap()
            .as_str()
            .trim_matches('"')
            .to_string(),
    )
    .join(
        serde_json::to_string(&model)
            .unwrap()
            .as_str()
            .trim_matches('"')
            .to_string(),
    )
    .join(
        serde_json::to_string(&input_type)
            .unwrap()
            .as_str()
            .trim_matches('"')
            .to_string(),
    )
    .join(
        serde_json::to_string(&replica_id)
            .unwrap()
            .as_str()
            .trim_matches('"')
            .to_string(),
    )
    .to_str()
    .unwrap()
    .to_string()
}

pub const OWNER: &str = "oes";
pub const DEFAULT_TOPIC_CHANNEL_SIZE: usize = 256;
pub const MAX_BATCH_SIZE_CLIP_TEXT: usize = 32;
