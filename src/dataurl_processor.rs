use data_url::DataUrl;

use crate::DataType;

pub fn data_type(data_url: &DataUrl) -> anyhow::Result<DataType> {
    let mime_type = data_url.mime_type().type_.as_str();
    let mime_subtype = data_url.mime_type().subtype.as_str();
    match (mime_type, mime_subtype) {
        ("text", "plain") => Ok(DataType::Text),
        ("image", "png") => Ok(DataType::Image),
        ("image", "jpeg") => Ok(DataType::Image),
        ("audio", "wav") => Ok(DataType::Audio),
        (t, st) => Err(anyhow::anyhow!("Unsupported media type: {}/{}", t, st)),
    }
}
