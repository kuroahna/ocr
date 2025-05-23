use std::ffi::CString;
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use axum_typed_multipart::TypedMultipart;
use image::{DynamicImage, EncodableLayout, ImageOutputFormat, Rgba};
use imageproc::rect::Rect;
use tesseract_plumbing::tesseract_sys::TessOcrEngineMode_OEM_LSTM_ONLY;
use tesseract_plumbing::TessBaseApi;
use prost::Message;
use proto::{
    AppliedFilter, AppliedFilters, ImageData, ImageMetadata, ImagePayload,
    LensOverlayClientContext, LensOverlayFilterType, LensOverlayObjectsRequest,
    LensOverlayRequestContext, LensOverlayRequestId, LensOverlayServerRequest,
    LensOverlayServerResponse, LocaleContext, Platform, Surface,
};
use rand::Rng;
use uuid::Uuid;

use crate::api::{
    BinarizeOperationRequest, CropOperationRequest, DrawFilledRectangleOperationRequest,
    GaussianBlurOperationRequest, MultipartRequest, OcrEngineRequest, OcrRequest, OperationRequest,
    OtsuBinarizeOperationRequest,
};
use crate::websocket::ServerStarted;

mod api;
mod proto;
mod websocket;

enum OcrEngine {
    Tesseract,
    GoogleLens,
}

impl From<OcrEngineRequest> for OcrEngine {
    fn from(value: OcrEngineRequest) -> Self {
        match value {
            OcrEngineRequest::Tesseract => Self::Tesseract,
            OcrEngineRequest::GoogleLens => Self::GoogleLens,
        }
    }
}

enum Operation {
    Invert,
    Binarize(BinarizeOperation),
    OtsuBinarize(OtsuBinarizeOperation),
    Crop(CropOperation),
    DrawFilledRectangle(DrawFilledRectangleOperation),
    GaussianBlur(GaussianBlurOperation),
}

impl From<OperationRequest> for Operation {
    fn from(value: OperationRequest) -> Self {
        match value {
            OperationRequest::Invert => Self::Invert,
            OperationRequest::Binarize(op) => Self::Binarize(BinarizeOperation::from(op)),
            OperationRequest::OtsuBinarize(op) => {
                Self::OtsuBinarize(OtsuBinarizeOperation::from(op))
            }
            OperationRequest::Crop(op) => Self::Crop(CropOperation::from(op)),
            OperationRequest::DrawFilledRectangle(op) => {
                Self::DrawFilledRectangle(DrawFilledRectangleOperation::from(op))
            }
            OperationRequest::GaussianBlur(op) => {
                Self::GaussianBlur(GaussianBlurOperation::from(op))
            }
        }
    }
}

struct BinarizeOperation {
    threshold: u8,
}

impl From<BinarizeOperationRequest> for BinarizeOperation {
    fn from(value: BinarizeOperationRequest) -> Self {
        Self {
            threshold: value.threshold,
        }
    }
}

struct OtsuBinarizeOperation {
    invert_threshold: bool,
}

impl From<OtsuBinarizeOperationRequest> for OtsuBinarizeOperation {
    fn from(value: OtsuBinarizeOperationRequest) -> Self {
        Self {
            invert_threshold: value.invert_threshold,
        }
    }
}

struct CropOperation {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl From<CropOperationRequest> for CropOperation {
    fn from(value: CropOperationRequest) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

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

impl From<DrawFilledRectangleOperationRequest> for DrawFilledRectangleOperation {
    fn from(value: DrawFilledRectangleOperationRequest) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
            r: value.r,
            g: value.g,
            b: value.b,
            a: value.a,
        }
    }
}

struct GaussianBlurOperation {
    sigma: f32,
}

impl From<GaussianBlurOperationRequest> for GaussianBlurOperation {
    fn from(value: GaussianBlurOperationRequest) -> Self {
        Self { sigma: value.sigma }
    }
}

struct Ocr {
    image: DynamicImage,
    ocr_engine: OcrEngine,
    operations: Vec<Operation>,
}

impl TryFrom<TypedMultipart<MultipartRequest>> for Ocr {
    type Error = String;

    fn try_from(value: TypedMultipart<MultipartRequest>) -> Result<Self, Self::Error> {
        let image = match image::load_from_memory(value.image.as_bytes()) {
            Ok(image) => image,
            Err(e) => return Err(e.to_string()),
        };
        let request: OcrRequest = match serde_json::from_str::<OcrRequest>(value.request.as_str()) {
            Ok(request) => request,
            Err(e) => return Err(e.to_string()),
        };
        let ocr_engine = OcrEngine::from(request.ocr_engine);
        let mut operations = Vec::new();
        for operation in request.operations {
            operations.push(Operation::from(operation));
        }

        Ok(Self {
            image,
            ocr_engine,
            operations,
        })
    }
}

struct ImageTransformer {
    image: DynamicImage,
}

impl ImageTransformer {
    fn new(image: DynamicImage) -> Self {
        Self { image }
    }

    fn image(&self) -> &DynamicImage {
        &self.image
    }

    fn transform(&mut self, operations: &[Operation]) {
        for operation in operations {
            match operation {
                Operation::Invert => {
                    println!("Inverting image");
                    self.image.invert();
                }
                Operation::Binarize(op) => {
                    println!("Binarizing image with threshold `{}`", op.threshold);
                    let mut grayscale_img = self.image.to_luma8();
                    imageproc::contrast::threshold_mut(&mut grayscale_img, op.threshold);
                    self.image = DynamicImage::ImageLuma8(grayscale_img);
                }
                Operation::OtsuBinarize(op) => {
                    let mut grayscale_img = self.image.to_luma8();
                    let otsu_level = imageproc::contrast::otsu_level(&grayscale_img);
                    let threshold_value = match op.invert_threshold {
                        true => 255 - otsu_level,
                        false => otsu_level,
                    };
                    println!(
                        "Otsu level found `{}` and binarizing image with threshold `{}`",
                        otsu_level, threshold_value
                    );
                    imageproc::contrast::threshold_mut(&mut grayscale_img, threshold_value);
                    self.image = DynamicImage::ImageLuma8(grayscale_img);
                }
                Operation::Crop(op) => {
                    println!(
                        "Cropping image with x `{}`, y `{}`, width `{}`, height `{}`",
                        op.x, op.y, op.width, op.height
                    );
                    self.image = self.image.crop(op.x, op.y, op.width, op.height);
                }
                Operation::DrawFilledRectangle(op) => {
                    println!(
                        "Drawing filled rectangle with x `{}`, y `{}`, width `{}`, height `{}`, r `{}`, g `{}`, b `{}`, a `{}`",
                        op.x, op.y, op.width, op.height, op.r, op.g, op.b, op.a
                    );
                    let out = imageproc::drawing::draw_filled_rect(
                        &self.image,
                        Rect::at(op.x, op.y).of_size(op.width, op.height),
                        Rgba([op.r, op.g, op.b, op.a]),
                    );
                    self.image = DynamicImage::ImageRgba8(out);
                }
                Operation::GaussianBlur(op) => {
                    println!("Gaussian blurring image with sigma `{}`", op.sigma);
                    let grayscale_img = self.image.to_luma8();
                    let out = imageproc::filter::gaussian_blur_f32(&grayscale_img, op.sigma);
                    self.image = DynamicImage::ImageLuma8(out);
                }
            }
        }
    }
}

struct TesseractOcrEngine {
    api: Mutex<TessBaseApi>,
}

impl TesseractOcrEngine {
    fn new(api: TessBaseApi) -> Self {
        Self {
            api: Mutex::new(api),
        }
    }

    fn ocr(&self, image: &DynamicImage) -> String {
        let mut lock = self.api.lock().unwrap();
        let bytes_per_pixel = image.color().bytes_per_pixel();
        let bytes_per_line = bytes_per_pixel as u32 * image.width();
        lock.set_image(
            image.as_bytes(),
            image.width().try_into().unwrap(),
            image.height().try_into().unwrap(),
            bytes_per_pixel.into(),
            bytes_per_line.try_into().unwrap(),
        )
        .unwrap();
        lock.get_utf8_text()
            .unwrap()
            .as_ref()
            .to_str()
            .unwrap()
            .to_owned()
    }
}

struct GoogleLensOcrEngine {
    client: reqwest::Client,
}

impl GoogleLensOcrEngine {
    fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn ocr(&self, image: &DynamicImage) -> Result<String, String> {
        let mut png_bytes = Cursor::new(Vec::new());
        image
            .write_to(&mut png_bytes, ImageOutputFormat::Png)
            .unwrap();

        let request = LensOverlayServerRequest {
            objects_request: Some(LensOverlayObjectsRequest {
                request_context: Some(LensOverlayRequestContext {
                    request_id: Some(LensOverlayRequestId {
                        uuid: Uuid::new_v4().as_u64_pair().0,
                        sequence_id: 1,
                        image_sequence_id: 1,
                        analytics_id: (0..16).map(|_| rand::rng().random()).collect(),
                        long_context_id: 1,
                        routing_info: None,
                    }),
                    client_context: Some(LensOverlayClientContext {
                        platform: Platform::Web.into(),
                        surface: Surface::Chromium.into(),
                        locale_context: Some(LocaleContext {
                            language: "ja".to_owned(),
                            region: "Asia/Tokyo".to_owned(),
                            // not set by chromium
                            time_zone: "".to_owned(),
                        }),
                        // not set by chromium
                        app_id: "".to_owned(),
                        client_filters: Some(AppliedFilters {
                            filter: vec![AppliedFilter {
                                filter_type: LensOverlayFilterType::AutoFilter.into(),
                                filter_payload: None,
                            }],
                        }),
                        rendering_context: None,
                        client_logging_data: None,
                    }),
                }),
                image_data: Some(ImageData {
                    payload: Some(ImagePayload {
                        image_bytes: png_bytes.into_inner(),
                    }),
                    // TODO: resize if width * height > 3_000_000
                    image_metadata: Some(ImageMetadata {
                        width: i32::try_from(image.width()).unwrap(),
                        height: i32::try_from(image.height()).unwrap(),
                    }),
                    significant_regions: Vec::new(),
                }),
                payload: None,
                viewport_request_context: None,
            }),
            interaction_request: None,
            client_logs: None,
        };

        let mut body = Vec::new();
        request.encode(&mut body).unwrap();

        match self
            .client
            .post("https://lensfrontend-pa.googleapis.com/v1/crupload")
            .header("Content-Type", "application/x-protobuf")
            .header("X-Goog-Api-Key", "AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY")
            .body(body)
            .send()
            .await
        {
            Ok(res) => {
                if !res.status().is_success() {
                    let error = format!(
                        "Request to Google Lens was not successful. Status code: `{}`. Response body: `{}`",
                        res.status().as_str(),
                        res.text().await.unwrap()
                    );
                    println!("{}", error);
                    return Err(error);
                }

                let response = res.bytes().await.unwrap();
                let response = LensOverlayServerResponse::decode(response).unwrap();

                let mut result = String::new();
                if let Some(response) = response.objects_response {
                    if let Some(text) = response.text {
                        if let Some(layout) = text.text_layout {
                            for paragraph in layout.paragraphs {
                                for line in paragraph.lines {
                                    for word in line.words {
                                        result.push_str(&word.plain_text);
                                        if let Some(separator) = word.text_separator {
                                            result.push_str(&separator);
                                        }
                                    }
                                }
                                result.push('\n');
                            }
                        }
                    }
                }

                Ok(result)
            }
            Err(e) => {
                let error = format!("Failed to send request to Google Lens: {}", e);
                println!("{}", error);
                Err(error)
            }
        }
    }
}

struct OcrEngineManager {
    tesseract: TesseractOcrEngine,
    google_lens: GoogleLensOcrEngine,
}

impl OcrEngineManager {
    fn new(tesseract: TesseractOcrEngine, google_lens: GoogleLensOcrEngine) -> Self {
        Self {
            tesseract,
            google_lens,
        }
    }

    async fn ocr(&self, ocr_engine: OcrEngine, image: &DynamicImage) -> Result<String, String> {
        match ocr_engine {
            OcrEngine::Tesseract => Ok(self.tesseract.ocr(image)),
            OcrEngine::GoogleLens => self.google_lens.ocr(image).await,
        }
    }
}

struct AppState {
    ocr_engine_manager: OcrEngineManager,
    websocket_server: ServerStarted,
}

impl AppState {
    fn new(ocr_engine_manager: OcrEngineManager, websocket_server: ServerStarted) -> Self {
        AppState {
            ocr_engine_manager,
            websocket_server,
        }
    }
}

async fn ocr(
    State(state): State<Arc<AppState>>,
    multipart: TypedMultipart<MultipartRequest>,
) -> impl IntoResponse {
    let ocr = match Ocr::try_from(multipart) {
        Ok(ocr) => ocr,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(e))
                .unwrap()
        }
    };

    let mut image_transformer = ImageTransformer::new(ocr.image);
    image_transformer.transform(ocr.operations.as_slice());
    let text = match state
        .ocr_engine_manager
        .ocr(ocr.ocr_engine, image_transformer.image())
        .await
    {
        Ok(text) => text,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(e))
                .unwrap()
        }
    };
    println!("{}", text);
    state.websocket_server.send_message(text);
    image_transformer.image().save("output.png").unwrap();

    Response::builder()
        .status(StatusCode::OK)
        .body(Body::empty())
        .unwrap()
}

#[tokio::main]
async fn main() {
    println!("Starting websocket server at `0.0.0.0:6677`");
    let websocket_server = websocket::Server::new(SocketAddr::from(([0, 0, 0, 0], 6677))).start();
    println!("Initializing Tesseract");
    let mut api = TessBaseApi::create();
    api.init_4(
        Some(CString::new("tessdata").unwrap().as_ref()),
        Some(CString::new("jpn").unwrap().as_ref()),
        TessOcrEngineMode_OEM_LSTM_ONLY,
    )
    .unwrap();
    let ocr_engine_manager =
        OcrEngineManager::new(TesseractOcrEngine::new(api), GoogleLensOcrEngine::new());
    let state = Arc::new(AppState::new(ocr_engine_manager, websocket_server));

    let app = Router::new()
        .route("/api/v1/ocr", post(ocr))
        .layer(DefaultBodyLimit::disable())
        .with_state(state);

    println!("Starting HTTP server at `0.0.0.0:9090`");
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], 9090)))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
