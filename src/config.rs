use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, EnumIter};

#[inline]
fn one() -> u32 {
    1
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Clone, Default)]
pub enum OesDevice {
    #[default]
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
    pub encodings: Vec<OesModelEncodingsConfig>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct OesModelEncodingsConfig {
    pub data_type: DataType,
    #[serde(default = "one")]
    pub replicas: u32,
    #[serde(default)]
    pub device: OesDevice,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Clone)]
pub enum DataType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "audio")]
    Audio,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash, Clone, EnumIter, AsRefStr, )]
pub enum OesModel {
    #[serde(rename = "openai/clip-vit-base-patch32")]
    #[strum(serialize = "openai/clip-vit-base-patch32")]
    OpenAIClip,
    #[serde(rename = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K")]
    #[strum(serialize = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K")]
    LaionClip,
    // #[serde(rename = "microsoft/phi-2")]
    // #[strum(serialize = "microsoft/phi-2")]
    // Phi2,
    #[serde(rename = "Alibaba-NLP/gte-Qwen1.5-7B-instruct")]
    #[strum(serialize = "Alibaba-NLP/gte-Qwen1.5-7B-instruct")]
    GteQwen,
}
