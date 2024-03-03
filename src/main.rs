use std::ffi::CString;
use std::io::Cursor;
use std::net::SocketAddr;
use std::str::from_utf8;
use std::sync::{Arc, OnceLock};

use axum::body::Body;
use axum::extract::{Multipart, State};
use axum::http::{Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use image::{DynamicImage, EncodableLayout, ImageOutputFormat, Rgba};
use imageproc::rect::Rect;
use regex::Regex;
use reqwest::multipart;
use serde::Deserialize;
use serde_json::Value;
use tesseract_plumbing::tesseract_sys::TessOcrEngineMode_OEM_LSTM_ONLY;
use tesseract_plumbing::TessBaseApi;
use tokio::sync::Mutex;

use crate::websocket::ServerStarted;

mod websocket;

static GOOGLE_LENS_OCR_RESPONSE_REGEX: OnceLock<Regex> = OnceLock::new();
static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

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
enum OcrEngine {
    #[serde(rename = "tesseract")]
    Tesseract,
    #[serde(rename = "googleLens")]
    GoogleLens,
}

#[derive(Deserialize)]
struct OcrRequest {
    #[serde(rename = "ocrEngine")]
    ocr_engine: OcrEngine,
    #[serde(rename = "operations")]
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

struct MultipartRequestBuilder {
    image: Option<Vec<u8>>,
    request: Option<String>,
}

impl MultipartRequestBuilder {
    fn new() -> Self {
        MultipartRequestBuilder {
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

    fn build(self) -> Result<MultipartRequest, String> {
        let image = match self.image {
            Some(image) => match image::load_from_memory(image.as_bytes()) {
                Ok(image) => image,
                Err(e) => return Err(e.to_string()),
            },
            None => return Err("image is missing".to_string()),
        };
        let request: OcrRequest = match self.request {
            Some(request) => match request.is_empty() {
                true => OcrRequest {
                    ocr_engine: OcrEngine::Tesseract,
                    operations: Vec::new(),
                },
                false => match serde_json::from_str(request.as_str()) {
                    Ok(request) => request,
                    Err(e) => return Err(e.to_string()),
                },
            },
            None => OcrRequest {
                ocr_engine: OcrEngine::Tesseract,
                operations: Vec::new(),
            },
        };
        Ok(MultipartRequest { image, request })
    }
}

struct MultipartRequest {
    image: DynamicImage,
    request: OcrRequest,
}

async fn ocr(
    State(state): State<Arc<Mutex<AppState>>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut builder = MultipartRequestBuilder::new();
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
    let multipart_request = match builder.build() {
        Ok(multipart_request) => multipart_request,
        Err(e) => {
            println!("{}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(e))
                .unwrap();
        }
    };

    let mut img = multipart_request.image;
    let request = multipart_request.request;
    for operation in request.operations {
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

    let text = match request.ocr_engine {
        OcrEngine::Tesseract => {
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
            utf8_text.as_ref().to_str().unwrap().to_string()
        }
        OcrEngine::GoogleLens => {
            let mut png_bytes = Cursor::new(Vec::new());
            img.write_to(&mut png_bytes, ImageOutputFormat::Png)
                .unwrap();
            let form_part = multipart::Part::bytes(png_bytes.into_inner())
                .file_name("image.png")
                .mime_str("image/png")
                .unwrap();
            let form = multipart::Form::new().part("encoded_image", form_part);
            match CLIENT
                .get()
                .expect("The client should be initialized upon startup")
                .post("https://lens.google.com/v3/upload")
                .multipart(form)
                .send()
                .await
            {
                Ok(res) => {
                    if !res.status().is_success() {
                        println!("Request to Google Lens was not successful");
                        return Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::empty())
                            .unwrap();
                    }

                    let text = res.text().await.unwrap();
                    let json5_str = GOOGLE_LENS_OCR_RESPONSE_REGEX
                        .get()
                        .expect("The regex should be initialized upon startup")
                        .captures(text.as_str())
                        .unwrap()
                        .get(1)
                        .unwrap()
                        .as_str();
                    let value: Value = json5::from_str(json5_str).unwrap();
                    let data = value
                        .get("data")
                        .unwrap()
                        .get(3)
                        .unwrap()
                        .get(4)
                        .unwrap()
                        .get(0)
                        .unwrap();
                    let mut result = String::new();
                    for lines in data.as_array().unwrap() {
                        for text in lines.as_array().unwrap() {
                            result.push_str(text.as_str().unwrap());
                        }
                    }
                    result
                }
                Err(e) => {
                    println!("Failed to send request to Google Lens: {}", e);
                    return Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .unwrap();
                }
            }
        }
    };

    {
        let locked_state = state.lock().await;
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

    img.save("output.png").unwrap();
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

    GOOGLE_LENS_OCR_RESPONSE_REGEX.get_or_init(|| {
        Regex::new(r">AF_initDataCallback\((\{key: 'ds:1'.*?)\);</script>").unwrap()
    });
    CLIENT.get_or_init(reqwest::Client::new);

    println!("Starting HTTP server at `0.0.0.0:9090`");
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], 9090)))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
