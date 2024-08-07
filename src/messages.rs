use image::DynamicImage;

use crate::DataType;

type TextType = String;
type ImageType = DynamicImage;
type AudioType = (Vec<f32>, u32);

#[derive(Debug, Clone)]
pub enum Input {
    Text(TextType),
    Image(ImageType),
    Audio(AudioType),
}

impl Input {
    pub fn data_type(&self) -> DataType {
        match self {
            Input::Text(_) => DataType::Text,
            Input::Image(_) => DataType::Image,
            Input::Audio(_) => DataType::Audio,
        }
    }
}

impl AsRef<Input> for Input {
    fn as_ref(&self) -> &Input {
        self
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingRequestHeader {
    pub rendezvous_topic: String,
    pub ordinal: u32,
}

#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub payload: Input,
    pub header: EmbeddingRequestHeader,
}

impl EmbeddingRequest {
    pub fn new(payload: Input, rendezvous_topic: String, ordinal: u32) -> Self {
        Self {
            payload,
            header: EmbeddingRequestHeader {
                rendezvous_topic,
                ordinal,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub values: Vec<f64>,
    pub ordinal: u32,
}

#[derive(Debug, Clone)]
pub enum EmbeddingMessage {
    Request(EmbeddingRequest),
    Response(EmbeddingResponse),
    Error(String),
}
