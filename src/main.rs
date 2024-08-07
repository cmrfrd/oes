use axum::extract::DefaultBodyLimit;
use bytes::Bytes;
use clap::{command, Parser, Subcommand};
use config::Config;
use std::{net::Ipv4Addr, time::Duration};
use strum::IntoEnumIterator;
use tabled::{builder::Builder, settings::Style};
use tokio::net::TcpListener;
use tower_http::{
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
    LatencyUnit,
};
use tracing::info;

use oes::{
    model_compatability_map, openai::server, Broker, DataType, EmbeddingMessage, OesBaseService,
    OesConfig, OesModelService, OesOaiService,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Run(RunCli),
    ListModels,
}

#[derive(Debug, Parser)]
pub struct RunCli {
    #[clap(long, default_value = "127.0.0.1")]
    host: Ipv4Addr,

    #[clap(long, default_value = "8080")]
    port: u16,

    #[clap(long)]
    model_config: Option<String>,
}

pub async fn start_server(addr: &str, args: RunCli) {
    tracing_subscriber::fmt().init();
    let oes_model_config = match args.model_config {
        Some(config_filename) => {
            match Config::builder()
                .add_source(config::File::with_name(&config_filename))
                .build()
            {
                Ok(config) => match config.try_deserialize::<OesConfig>() {
                    Ok(config) => config,
                    Err(e) => {
                        panic!("Error parsing config: {}", e);
                    }
                },
                Err(e) => {
                    panic!("Error reading config filename {}: {}", config_filename, e);
                }
            }
        }
        None => OesConfig { models: vec![] },
    };

    info!("Starting OES Broker");
    let broker = Broker::<EmbeddingMessage>::new();

    info!("Starting OES Models");
    let oes_model_service = OesModelService::new(broker.clone(), oes_model_config.clone());
    oes_model_service.run();

    info!("Starting OES API");
    let oes_base = OesBaseService::new();
    let oes_base_service = OesBaseService::router(oes_base);
    let oes_oai = OesOaiService::new(broker.clone(), oes_model_config.clone());
    let oes_oai_service = server::new(oes_oai);
    let app = axum::Router::new()
        // .layer(DefaultBodyLimit::max(1000 * 1000 * 1000))
        .layer(DefaultBodyLimit::disable())
        // .layer(RequestBodyLimitLayer::new(1000 * 1000 * 1000)) // 1GB
        // .layer(extractor_middleware::<ContentLengthLimit<(), 1024>>())
        .layer(
            TraceLayer::new_for_http()
                .on_body_chunk(|chunk: &Bytes, latency: Duration, _: &tracing::Span| {
                    tracing::trace!(size_bytes = chunk.len(), latency = ?latency, "sending body chunk")
                })
                .make_span_with(DefaultMakeSpan::new().include_headers(true))
                .on_response(DefaultOnResponse::new().include_headers(true).latency_unit(LatencyUnit::Micros)),
        )
        // All requests that prefix /oai will go here
        .layer(DefaultBodyLimit::disable())
        .nest("/", oes_oai_service)
        .layer(DefaultBodyLimit::disable())
        // All other requests will go here
        .nest("/", oes_base_service);

    let listener = TcpListener::bind(addr).await.unwrap();
    info!("Listening on: {}", addr);
    axum::serve(listener, app).await.unwrap();
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(cli_args) => {
            let addr = format!("{}:{}", cli_args.host, cli_args.port);
            start_server(&addr, cli_args).await;
        }
        Commands::ListModels => {
            let res = model_compatability_map();
            let base_col = vec!["Model ID".to_string()];
            let data_type_cols = DataType::iter()
                .map(|model| model.as_ref().to_string())
                .collect::<Vec<_>>();
            let all_cols = base_col
                .iter()
                .chain(data_type_cols.iter())
                .map(|s| s.clone())
                .collect::<Vec<_>>();

            let mut table_data = vec![all_cols];
            let mut kvs = res.iter().collect::<Vec<_>>();
            kvs.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));
            for (model, data_types) in kvs {
                let mut row = vec![model.as_ref().to_string()];
                for data_type in DataType::iter() {
                    if data_types.contains(&data_type) {
                        row.push("X".to_string());
                    } else {
                        row.push("".to_string());
                    }
                }
                table_data.push(row);
            }

            let mut table = Builder::from_iter(table_data).build();
            table.with(Style::rounded());
            println!("{}", table);
        }
    }
}
