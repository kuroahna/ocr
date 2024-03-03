use std::net::SocketAddr;

use clap::{arg, Parser};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use smol::net::TcpListener;
use smol_hyper::rt::{FuturesIo, SmolTimer};

use crate::websocket::ServerStarted;

mod websocket;

/// A WebSocket server that receives messages via HTTP and broadcasts it to all
/// the connected WebSocket clients
///
/// Example:
///     curl localhost:<HTTP_PORT> -d "the message"
#[derive(Parser)]
#[command(name = "websocket")]
#[command(version, about, verbatim_doc_comment)]
struct Cli {
    /// The port to bind to for the WebSocket server
    #[arg(long, default_value_t = 6677)]
    websocket_port: u16,
    /// The port to bind to for the HTTP server
    #[arg(long, default_value_t = 9090)]
    http_port: u16,
}

async fn handle(
    request: Request<hyper::body::Incoming>,
    websocket_server: &ServerStarted,
) -> hyper::http::Result<Response<Full<Bytes>>> {
    let body = match request.collect().await {
        Ok(body) => body.to_bytes(),
        Err(e) => {
            println!("Failed to read request body: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::default());
        }
    };
    let text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(e) => {
            println!("Request body is not valid UTF-8: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::default());
        }
    };
    println!("Sending `{}` to all connected WebSocket clients", text);
    websocket_server.send_message(text);
    Response::builder()
        .status(StatusCode::OK)
        .body(Full::default())
}

fn main() {
    let cli = Cli::parse();

    println!(
        "Starting WebSocket server at `0.0.0.0:{}`",
        cli.websocket_port
    );
    let websocket_server =
        websocket::Server::new(SocketAddr::from(([0, 0, 0, 0], cli.websocket_port))).start();

    let executor = smol::LocalExecutor::new();
    smol::block_on(async {
        println!("Starting HTTP server at `0.0.0.0:{}`", cli.http_port);
        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], cli.http_port)))
            .await
            .unwrap();
        executor
            .run(async {
                loop {
                    let (stream, _) = listener.accept().await.unwrap();
                    let io = FuturesIo::new(stream);
                    executor
                        .spawn(async {
                            http1::Builder::new()
                                .timer(SmolTimer::new())
                                .serve_connection(
                                    io,
                                    service_fn(|req| handle(req, &websocket_server)),
                                )
                                .await
                                .unwrap();
                        })
                        .detach();
                }
            })
            .await;
    });
}
