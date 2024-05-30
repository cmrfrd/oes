use axum::extract::DefaultBodyLimit;
use bytes::Bytes;
use clap::{command, Parser, Subcommand};
use std::{net::Ipv4Addr, time::Duration};
use tokio::net::TcpListener;
use tower_http::{
    limit::RequestBodyLimitLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
    LatencyUnit,
};

use oes::{
    openai::server, Broker, EmbeddingMessage, OesBaseService, OesConfig, OesModelService,
    OesOaiService,
};

use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Run(RunCli),
}

#[derive(Debug, Parser)]
pub struct RunCli {
    #[clap(long, default_value = "127.0.0.1")]
    host: Ipv4Addr,

    #[clap(long, default_value = "8080")]
    port: u16,

    #[clap(long)]
    config: Option<String>,
}

pub async fn start_server(addr: &str, args: RunCli) {
    tracing_subscriber::fmt().init();
    let oes_config = match args.config {
        Some(config) => match std::fs::read_to_string(config.clone()) {
            Ok(config) => match serde_json::from_str::<OesConfig>(&config) {
                Ok(config) => config,
                Err(e) => {
                    panic!("Error parsing config: {}", e);
                }
            },
            Err(e) => {
                panic!("Error reading config {}: {}", config, e);
            }
        },
        None => OesConfig { models: vec![] },
    };

    info!("Starting OES Broker");
    let broker = Broker::<EmbeddingMessage>::new();

    info!("Starting OES Models");
    let oes_model_service = OesModelService::new(broker.clone(), oes_config.clone());
    oes_model_service.run();

    info!("Starting OES API");
    let oes_base = OesBaseService::new();
    let oes_base_service = OesBaseService::router(oes_base);
    let oes_oai = OesOaiService::new(broker.clone(), oes_config.clone());
    let oes_oai_service = server::new(oes_oai);
    let app = axum::Router::new()
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(50 * 1000 * 1000))
        .layer(
            TraceLayer::new_for_http()
                .on_body_chunk(|chunk: &Bytes, latency: Duration, _: &tracing::Span| {
                    tracing::trace!(size_bytes = chunk.len(), latency = ?latency, "sending body chunk")
                })
                .make_span_with(DefaultMakeSpan::new().include_headers(true))
                .on_response(DefaultOnResponse::new().include_headers(true).latency_unit(LatencyUnit::Micros)),
        )
        // All requests that prefix /oai will go here
        .nest("/", oes_oai_service)
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
    }
}
