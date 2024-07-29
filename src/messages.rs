#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: String,
    pub rendezvous_topic: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum EmbeddingMessage {
    Request(EmbeddingRequest),
    Response(EmbeddingResponse),
    Error(String),
}
