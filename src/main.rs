use std::ffi::CString;
use std::net::SocketAddr;
use std::str::from_utf8;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Multipart, State};
use axum::http::{Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use image::{DynamicImage, EncodableLayout, Rgba};
use imageproc::rect::Rect;
use serde::Deserialize;
use tesseract_plumbing::tesseract_sys::TessOcrEngineMode_OEM_LSTM_ONLY;
use tesseract_plumbing::TessBaseApi;
use tokio::sync::Mutex;

use crate::websocket::ServerStarted;

mod websocket;

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Operation {
    #[serde(rename = "invert")]
    Invert(InvertOperation),
    #[serde(rename = "binarize")]
    Binarize(BinarizeOperation),
    #[serde(rename = "otsuBinarize")]
    OtsuBinarize(OtsuBinarizeOperation),
    #[serde(rename = "crop")]
    Crop(CropOperation),
    #[serde(rename = "drawFilledRectangle")]
    DrawFilledRectangle(DrawFilledRectangleOperation),
    #[serde(rename = "gaussianBlur")]
    GaussianBlur(GaussianBlurOperation),
}

#[derive(Deserialize)]
struct InvertOperation {}

#[derive(Deserialize)]
struct BinarizeOperation {
    threshold: u8,
}

#[derive(Deserialize)]
struct OtsuBinarizeOperation {
    #[serde(rename = "invertThreshold")]
    invert_threshold: Option<bool>,
}

#[derive(Deserialize)]
struct CropOperation {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct DrawFilledRectangleOperation {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[derive(Deserialize)]
struct GaussianBlurOperation {
    #[serde(rename = "sigma")]
    sigma: f32,
}

#[derive(Deserialize)]
struct Operations {
    operations: Vec<Operation>,
}

enum MultipartRequestFormField {
    Image,
    Request,
}

impl TryFrom<&str> for MultipartRequestFormField {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "image" => Ok(MultipartRequestFormField::Image),
            "request" => Ok(MultipartRequestFormField::Request),
            _ => Err(format!("Unknown multipart form field name `{}`", value)),
        }
    }
}

struct RequestBuilder {
    image: Option<Vec<u8>>,
    request: Option<String>,
}

impl RequestBuilder {
    fn new() -> Self {
        RequestBuilder {
            image: None,
            request: None,
        }
    }

    fn image(&mut self, image: Vec<u8>) -> &mut Self {
        self.image = Some(image);
        self
    }

    fn request(&mut self, request: String) -> &mut Self {
        self.request = Some(request);
        self
    }

    fn build(self) -> Result<TheRequest, String> {
        let image = match self.image {
            Some(image) => match image::load_from_memory(image.as_bytes()) {
                Ok(image) => image,
                Err(e) => return Err(e.to_string()),
            },
            None => return Err("image is missing".to_string()),
        };
        let request: Operations = match self.request {
            Some(request) => match request.is_empty() {
                true => Operations {
                    operations: Vec::new(),
                },
                false => match serde_json::from_str(request.as_str()) {
                    Ok(request) => request,
                    Err(e) => return Err(e.to_string()),
                },
            },
            None => Operations {
                operations: Vec::new(),
            },
        };
        Ok(TheRequest { image, request })
    }
}

struct TheRequest {
    image: DynamicImage,
    request: Operations,
}

async fn ocr(
    State(state): State<Arc<Mutex<AppState>>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut builder = RequestBuilder::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        let value = field.bytes().await.unwrap();

        let key = match MultipartRequestFormField::try_from(name.as_str()) {
            Ok(key) => key,
            Err(_) => continue,
        };
        match key {
            MultipartRequestFormField::Image => builder.image(value.to_vec()),
            MultipartRequestFormField::Request => {
                builder.request(from_utf8(value.as_bytes()).unwrap().to_string())
            }
        };
    }
    let the_request = match builder.build() {
        Ok(the_request) => the_request,
        Err(e) => {
            println!("{}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(e))
                .unwrap();
        }
    };

    let mut img = the_request.image;
    let operations = the_request.request;
    for operation in operations.operations {
        match operation {
            Operation::Invert(_) => {
                println!("Inverting image");
                img.invert();
            }
            Operation::Binarize(op) => {
                println!("Binarizing image with threshold `{}`", op.threshold);
                let mut grayscale_img = img.to_luma8();
                imageproc::contrast::threshold_mut(&mut grayscale_img, op.threshold);
                img = DynamicImage::ImageLuma8(grayscale_img);
            }
            Operation::OtsuBinarize(op) => {
                let invert_threshold = op.invert_threshold.unwrap_or(false);
                let mut grayscale_img = img.to_luma8();
                let otsu_level = imageproc::contrast::otsu_level(&grayscale_img);
                let threshold_value = match invert_threshold {
                    true => 255 - otsu_level,
                    false => otsu_level,
                };
                println!(
                    "Otsu level found `{}` and binarizing image with threshold `{}`",
                    otsu_level, threshold_value
                );
                imageproc::contrast::threshold_mut(&mut grayscale_img, threshold_value);
                img = DynamicImage::ImageLuma8(grayscale_img);
            }
            Operation::Crop(op) => {
                println!(
                    "Cropping image with x `{}`, y `{}`, width `{}`, height `{}`",
                    op.x, op.y, op.width, op.height
                );
                img = img.crop(op.x, op.y, op.width, op.height);
            }
            Operation::DrawFilledRectangle(op) => {
                println!(
                    "Drawing filled rectangle with x `{}`, y `{}`, width `{}`, height `{}`, r `{}`, g `{}`, b `{}`, a `{}`",
                    op.x, op.y, op.width, op.height, op.r, op.g, op.b, op.a
                );
                let out = imageproc::drawing::draw_filled_rect(
                    &img,
                    Rect::at(op.x, op.y).of_size(op.width, op.height),
                    Rgba([op.r, op.g, op.b, op.a]),
                );
                img = DynamicImage::ImageRgba8(out);
            }
            Operation::GaussianBlur(op) => {
                println!("Gaussian blurring image with sigma `{}`", op.sigma);
                let grayscale_img = img.to_luma8();
                let out = imageproc::filter::gaussian_blur_f32(&grayscale_img, op.sigma);
                img = DynamicImage::ImageLuma8(out);
            }
        }
    }
    let img = img.to_rgb8();
    img.save("output.png").unwrap();

    {
        let mut locked_state = state.lock().await;
        locked_state
            .tesseract_api
            .set_image(
                img.as_bytes(),
                img.width().try_into().unwrap(),
                img.height().try_into().unwrap(),
                3,
                (3 * img.width()).try_into().unwrap(),
            )
            .unwrap();
        let utf8_text = locked_state.tesseract_api.get_utf8_text().unwrap();
        let text = utf8_text.as_ref().to_str().unwrap();
        let whitespace_removed: String = text
            .lines()
            .map(|line| {
                line.chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
            })
            .collect::<Vec<String>>()
            .join("\n");
        println!("{}\n", whitespace_removed);
        locked_state
            .websocket_server
            .send_message(whitespace_removed);
    }

    Response::builder()
        .status(StatusCode::OK)
        .body(Body::empty())
        .unwrap()
}

struct AppState {
    tesseract_api: TessBaseApi,
    websocket_server: ServerStarted,
}

impl AppState {
    fn new(tesseract_api: TessBaseApi, websocket_server: ServerStarted) -> Self {
        AppState {
            tesseract_api,
            websocket_server,
        }
    }
}

#[tokio::main]
async fn main() {
    println!("Starting websocket server at `0.0.0.0:6677`");
    let server = websocket::Server::new(SocketAddr::from(([0, 0, 0, 0], 6677))).start();
    println!("Initializing Tesseract");
    let mut api = TessBaseApi::create();
    api.init_4(
        Some(CString::new("tessdata").unwrap().as_ref()),
        Some(CString::new("jpn").unwrap().as_ref()),
        TessOcrEngineMode_OEM_LSTM_ONLY,
    )
    .unwrap();
    let state = Arc::new(Mutex::new(AppState::new(api, server)));

    let app = Router::new()
        .route("/api/v1/ocr", post(ocr))
        .with_state(state);

    println!("Starting HTTP server at `0.0.0.0:9090`");
    let addr = SocketAddr::from(([0, 0, 0, 0], 9090));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
