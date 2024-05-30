use candle_core::Device;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Clone)]
pub enum OesDevice {
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "gpu")]
    Gpu,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum OesAction {
    #[serde(rename = "embed")]
    Embed,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum OesVerb {
    #[serde(rename = "request")]
    Request,
    #[serde(rename = "response")]
    Response,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OesConfig {
    #[serde(default)]
    pub models: Vec<OesModelConfig>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct OesModelConfig {
    pub model_name: OesModel,
    pub input_types: InputType,
    pub replicas: u32,
    pub device: OesDevice,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Clone)]
pub enum InputType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Clone)]
pub enum OesModel {
    #[serde(rename = "clip-vit-base-patch32")]
    OpenAIClip,
}
