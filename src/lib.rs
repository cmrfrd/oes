#![feature(const_trait_impl)]

pub mod broker;
pub mod config;
pub mod dataurl_processor;
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

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;
use symphonia::core::io::MediaSource;

pub const OWNER: &str = "oes";
pub const DEFAULT_TOPIC_CHANNEL_SIZE: usize = 256;
pub const MAX_BATCH_SIZE: usize = 32;

pub const BAD_REQUEST: &str = "Bad Request";
pub const FOUR_HUNDRED: &str = "400";

// Update this map with more model support
#[inline]
pub fn model_compatability_map() -> HashMap<OesModel, Vec<DataType>> {
    let mut map = HashMap::new();
    map.insert(OesModel::OpenAIClip, vec![DataType::Text, DataType::Image]);
    map.insert(OesModel::LaionClip, vec![DataType::Text, DataType::Image]);
    map.insert(OesModel::GteQwen, vec![DataType::Text]);
    map.insert(OesModel::Whisper, vec![DataType::Audio]);
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
    input_type: DataType,
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

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfo {
    model: OesModel,
    data_type: DataType,
}

impl FromStr for ModelInfo {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (model, data_type) = s
            .rsplit_once('/')
            .context("Invalid model string format. Expected model_id/input_format")?;
        Ok(ModelInfo {
            model: serde_json::from_value(serde_json::Value::String(model.to_string()))?,
            data_type: serde_json::from_value(serde_json::Value::String(data_type.to_string()))?,
        })
    }
}

pub(crate) fn conv<T>(
    samples: &mut Vec<f32>,
    data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>,
) where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

pub(crate) fn _pcm_decode<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<(Vec<f32>, u32)> {
    // Open the media source.
    let src = std::fs::File::open(path)?;
    pcm_decode_raw(Box::new(src))
}

pub(crate) fn pcm_decode_raw(
    media_source: Box<dyn MediaSource>,
) -> anyhow::Result<(Vec<f32>, u32)> {
    // Create the media source stream.
    let mss = symphonia::core::io::MediaSourceStream::new(media_source, Default::default());

    // Create a probe hint using the file's extension. [Optional]
    let hint = symphonia::core::probe::Hint::new();

    // Use the default options for metadata and format readers.
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    // The decode loop.
    while let Ok(packet) = format.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}
