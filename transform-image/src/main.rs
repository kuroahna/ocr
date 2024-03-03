use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use image::{DynamicImage, ImageFormat, Rgba};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use imageproc::rect::Rect;

/// A tool to transform images
///
/// This is useful for OCR where the image typically needs to be pre-processed
/// for best results
#[derive(Parser)]
#[command(name = "transform-image")]
#[command(version, about, verbatim_doc_comment)]
struct Cli {
    #[command(subcommand)]
    operation: Operation,
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
enum Operation {
    /// Invert the colors of the image
    #[command(verbatim_doc_comment)]
    Invert,
    /// Binarize the image, resulting in a grayscale image
    Binarize(BinarizeOperation),
    /// Binarize the image automatically using Otsu's method, resulting in a
    /// grayscale image
    OtsuBinarize(OtsuBinarizeOperation),
    /// Crop the image
    Crop(CropOperation),
    /// Draw a filled rectangle on the image
    DrawFilledRectangle(DrawFilledRectangleOperation),
    /// Blur the image using a Gaussian of standard deviation sigma
    GaussianBlur(GaussianBlurOperation),
}

#[derive(Args)]
struct BinarizeOperation {
    /// Pixels with intensity equal to the threshold are assigned to the background
    ///
    /// Valid values are from 0 to 255 (inclusive)
    #[arg(long)]
    threshold: u8,
}

#[derive(Args)]
struct OtsuBinarizeOperation {
    /// Enable to invert the threshold returned by Otsu's method
    ///
    /// This is useful if the colors of the image have been inverted
    #[arg(long, default_value_t = false)]
    invert_threshold: bool,
}

#[derive(Args)]
struct CropOperation {
    /// The x coordinate of the bounding box
    #[arg(long)]
    x: u32,
    /// The y coordinate of the bounding box
    #[arg(long)]
    y: u32,
    /// The width of the bounding box
    #[arg(long)]
    width: u32,
    /// The height of the bounding box
    #[arg(long)]
    height: u32,
}

#[derive(Args)]
struct DrawFilledRectangleOperation {
    /// The x coordinate of the bounding box
    #[arg(long)]
    x: i32,
    /// The y coordinate of the bounding box
    #[arg(long)]
    y: i32,
    /// The width of the bounding box
    #[arg(long)]
    width: u32,
    /// The height of the bounding box
    #[arg(long)]
    height: u32,
    /// The amount of red to apply
    ///
    /// Valid values are from 0 to 255 (inclusive)
    #[arg(long)]
    r: u8,
    /// The amount of green to apply
    ///
    /// Valid values are from 0 to 255 (inclusive)
    #[arg(long)]
    g: u8,
    /// The amount of blue to apply
    ///
    /// Valid values are from 0 to 255 (inclusive)
    #[arg(long)]
    b: u8,
    /// The amount of alpha to apply
    ///
    /// Valid values are from 0 to 255 (inclusive)
    #[arg(long)]
    a: u8,
}

#[derive(Args)]
struct GaussianBlurOperation {
    /// The Gaussian standard deviation sigma to blur the image. The kernel used
    /// has type f32 and all intermediate calculations are performed at this
    /// type
    ///
    /// Valid values are sigma > 0
    #[arg(long)]
    sigma: f32,
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
    let format = image::guess_format(buffer.as_slice()).unwrap();
    let mut img = image::load_from_memory_with_format(buffer.as_slice(), format).unwrap();

    match &cli.operation {
        Operation::Invert => {
            cli.run_if_debug_enabled(|| println!("Inverting image"));
            img.invert();
        }
        Operation::Binarize(op) => {
            cli.run_if_debug_enabled(|| {
                println!("Binarizing image with threshold `{}`", op.threshold)
            });
            let mut grayscale_img = img.to_luma8();
            imageproc::contrast::threshold_mut(&mut grayscale_img, op.threshold);
            img = DynamicImage::ImageLuma8(grayscale_img);
        }
        Operation::OtsuBinarize(op) => {
            let invert_threshold = op.invert_threshold;
            let mut grayscale_img = img.to_luma8();
            let otsu_level = imageproc::contrast::otsu_level(&grayscale_img);
            let threshold_value = match invert_threshold {
                true => 255 - otsu_level,
                false => otsu_level,
            };
            cli.run_if_debug_enabled(|| {
                println!(
                    "Otsu level found `{}` and binarizing image with threshold `{}`",
                    otsu_level, threshold_value
                )
            });
            imageproc::contrast::threshold_mut(&mut grayscale_img, threshold_value);
            img = DynamicImage::ImageLuma8(grayscale_img);
        }
        Operation::Crop(op) => {
            cli.run_if_debug_enabled(|| {
                println!(
                    "Cropping image with x `{}`, y `{}`, width `{}`, height `{}`",
                    op.x, op.y, op.width, op.height
                )
            });
            img = img.crop(op.x, op.y, op.width, op.height);
        }
        Operation::DrawFilledRectangle(op) => {
            cli.run_if_debug_enabled(|| println!(
                "Drawing filled rectangle with x `{}`, y `{}`, width `{}`, height `{}`, r `{}`, g `{}`, b `{}`, a `{}`",
                op.x, op.y, op.width, op.height, op.r, op.g, op.b, op.a
            ));
            let out = imageproc::drawing::draw_filled_rect(
                &img,
                Rect::at(op.x, op.y).of_size(op.width, op.height),
                Rgba([op.r, op.g, op.b, op.a]),
            );
            img = DynamicImage::ImageRgba8(out);
        }
        Operation::GaussianBlur(op) => {
            cli.run_if_debug_enabled(|| {
                println!("Gaussian blurring image with sigma `{}`", op.sigma)
            });
            let grayscale_img = img.to_luma8();
            let out = imageproc::filter::gaussian_blur_f32(&grayscale_img, op.sigma);
            img = DynamicImage::ImageLuma8(out);
        }
    };

    if let Some(path) = &cli.output {
        cli.run_if_debug_enabled(|| {
            println!("Writing transformed image to `{}`", path.to_str().unwrap())
        });
        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);
        img.write_to(&mut writer, format).unwrap();
    } else {
        cli.run_if_debug_enabled(|| println!("Writing transformed image to STDOUT"));

        // Seek is not implemented for stdout
        // https://github.com/rust-lang/rust/issues/72802
        //
        // img.write_to(&mut std::io::stdout(), format).unwrap();
        match format {
            ImageFormat::Png => img
                .write_with_encoder(PngEncoder::new(std::io::stdout()))
                .unwrap(),
            ImageFormat::Jpeg => img
                .write_with_encoder(JpegEncoder::new(std::io::stdout()))
                .unwrap(),
            _ => panic!(
                "Could not write transformed image to STDOUT. Unknown format: `{:?}`",
                format
            ),
        }
    }
}
