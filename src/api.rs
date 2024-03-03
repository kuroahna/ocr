use axum_typed_multipart::TryFromMultipart;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum OperationRequest {
    #[serde(rename = "invert")]
    Invert,
    #[serde(rename = "binarize")]
    Binarize(BinarizeOperationRequest),
    #[serde(rename = "otsuBinarize")]
    OtsuBinarize(OtsuBinarizeOperationRequest),
    #[serde(rename = "crop")]
    Crop(CropOperationRequest),
    #[serde(rename = "drawFilledRectangle")]
    DrawFilledRectangle(DrawFilledRectangleOperationRequest),
    #[serde(rename = "gaussianBlur")]
    GaussianBlur(GaussianBlurOperationRequest),
}

#[derive(Deserialize)]
pub struct BinarizeOperationRequest {
    pub threshold: u8,
}

#[derive(Deserialize)]
pub struct OtsuBinarizeOperationRequest {
    #[serde(rename = "invertThreshold", default)]
    pub invert_threshold: bool,
}

#[derive(Deserialize)]
pub struct CropOperationRequest {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Deserialize)]
pub struct DrawFilledRectangleOperationRequest {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Deserialize)]
pub struct GaussianBlurOperationRequest {
    pub sigma: f32,
}

#[derive(Deserialize)]
pub enum OcrEngineRequest {
    #[serde(rename = "tesseract")]
    Tesseract,
    #[serde(rename = "googleLens")]
    GoogleLens,
}

#[derive(Deserialize)]
pub struct OcrRequest {
    #[serde(rename = "ocrEngine")]
    pub ocr_engine: OcrEngineRequest,
    pub operations: Vec<OperationRequest>,
}

#[derive(TryFromMultipart)]
pub struct MultipartRequest {
    pub image: axum::body::Bytes,
    pub request: String,
}
