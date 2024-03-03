use std::ffi::CString;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use image::ImageOutputFormat;
use regex::Regex;
use serde_json::Value;
use tesseract_plumbing::tesseract_sys::TessOcrEngineMode_OEM_LSTM_ONLY;
use tesseract_plumbing::TessBaseApi;

const GOOGLE_LENS_REGEX_PATTERN: &str = r">AF_initDataCallback\((\{key: 'ds:1'.*?)\);</script>";

/// An OCR tool that reads an image from the input source and writes the
/// recognized text to the output destination
#[derive(Parser)]
#[command(name = "ocr")]
#[command(version, about, verbatim_doc_comment)]
struct Cli {
    #[command(subcommand)]
    ocr_engine: OcrEngine,
    /// The path to the input file. If not set, STDIN is used instead
    #[arg(long, value_name = "INPUT_FILE_PATH", default_value = None)]
    input: Option<PathBuf>,
    /// The path to the output file. If not set, STDOUT is used instead
    #[arg(long, value_name = "OUTPUT_FILE_PATH", default_value = None)]
    output: Option<PathBuf>,
    /// Enables debug logging
    #[arg(long, default_value_t = false)]
    debug: bool,
}

impl Cli {
    fn run_if_debug_enabled<T: Default, F: FnOnce() -> T>(&self, op: F) -> T {
        if self.debug {
            op()
        } else {
            T::default()
        }
    }
}

#[derive(Subcommand)]
enum OcrEngine {
    /// Uses Tesseract for the OCR engine (local)
    ///
    /// This is the fastest OCR option because it runs locally on your machine
    /// but may not be as accurate. Tesseract works well for images that have
    /// black text and a white background with standard fonts
    ///
    /// See https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html on how
    /// to pre-process the image for better results
    #[command(verbatim_doc_comment)]
    Tesseract {
        /// The path to the tessdata directory
        #[arg(long, default_value = "tessdata")]
        tessdata_dir: String,
        /// The language model to use
        #[arg(long, default_value = "jpn")]
        language: String,
    },
    /// Uses Google Lens for the OCR engine (remote)
    ///
    /// This is generally slower because it makes a network request and uploads
    /// the image to Google to process the image, but is the most accurate.
    /// Pre-processing the image is usually not necessary but can help with the
    /// accuracy even further. It can also reduce the image size which will make
    /// uploading to Google and processing the image even faster
    ///
    /// See https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html on how
    /// to pre-process the image for better results
    #[command(verbatim_doc_comment)]
    GoogleLens,
}

fn main() {
    let cli = Cli::parse();

    let mut buffer = Vec::new();
    if let Some(path) = &cli.input {
        cli.run_if_debug_enabled(|| println!("Reading image from `{}`", path.to_str().unwrap()));
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);
        reader.read_to_end(&mut buffer).unwrap();
    } else {
        cli.run_if_debug_enabled(|| println!("Reading image from STDIN"));
        std::io::stdin().read_to_end(&mut buffer).unwrap();
    }
    let img = image::load_from_memory(buffer.as_slice()).unwrap();

    let text = match &cli.ocr_engine {
        OcrEngine::Tesseract {
            tessdata_dir,
            language,
        } => {
            cli.run_if_debug_enabled(|| println!("Initializing Tesseract"));
            let mut api = TessBaseApi::create();
            api.init_4(
                Some(CString::new(tessdata_dir.as_bytes()).unwrap().as_ref()),
                Some(CString::new(language.as_bytes()).unwrap().as_ref()),
                TessOcrEngineMode_OEM_LSTM_ONLY,
            )
            .unwrap();
            cli.run_if_debug_enabled(|| println!("Initialized Tesseract"));
            api.set_image(
                img.as_bytes(),
                img.width().try_into().unwrap(),
                img.height().try_into().unwrap(),
                3,
                (3 * img.width()).try_into().unwrap(),
            )
            .unwrap();
            let text = api
                .get_utf8_text()
                .unwrap()
                .as_ref()
                .to_str()
                .unwrap()
                .to_owned();
            cli.run_if_debug_enabled(|| println!("Tesseract returned the text: `{}`", text));
            text
        }
        OcrEngine::GoogleLens => {
            cli.run_if_debug_enabled(|| println!("Initializing Google Lens"));
            let mut png_bytes = Cursor::new(Vec::new());
            img.write_to(&mut png_bytes, ImageOutputFormat::Png)
                .unwrap();
            let client = reqwest::blocking::Client::new();
            let form_part = reqwest::blocking::multipart::Part::bytes(png_bytes.into_inner())
                .file_name("image.png")
                .mime_str("image/png")
                .unwrap();
            let form = reqwest::blocking::multipart::Form::new().part("encoded_image", form_part);
            cli.run_if_debug_enabled(|| println!("Initialized Google Lens"));
            match client
                .post("https://lens.google.com/v3/upload")
                .multipart(form)
                .send()
            {
                Ok(res) => {
                    if !res.status().is_success() {
                        panic!("Request to Google Lens was not successful. Got response status code `{}` with body `{}`", res.status(), res.text().unwrap());
                    }

                    let regex = Regex::new(GOOGLE_LENS_REGEX_PATTERN).unwrap();
                    let text = res.text().unwrap();
                    let json5_str = regex
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
                    panic!("Failed to send request to Google Lens: {}", e);
                }
            }
        }
    };

    if let Some(path) = &cli.output {
        cli.run_if_debug_enabled(|| println!("Writing `{}` to `{}`", text, path.to_str().unwrap()));
        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);
        writer.write_all(text.as_bytes()).unwrap();
    } else {
        cli.run_if_debug_enabled(|| println!("Writing `{}` to STDOUT", text));
        std::io::stdout().write_all(text.as_bytes()).unwrap();
    }
}
