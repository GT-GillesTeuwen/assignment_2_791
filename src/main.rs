use image::{GenericImageView, GrayImage, Luma};
use rfd::FileDialog;
use std::fs;
use std::path::{Path, PathBuf};

mod file_writing;
mod histogram_drawer;
mod kapur;
mod otsu;
mod stats;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Directory containing the images
    println!("Select the folder containing the images");
    let img_dir = FileDialog::new()
        .set_directory("/")
        .pick_folder()
        .expect("No folder selected")
        .to_str()
        .expect("Failed to convert path to string")
        .to_string();
    // Create necessary directories
    let dirs = [
        "results/otsu/exhaustive",
        "results/otsu/sa/",
        "results/otsu/vns/",
        "results/kapur/exhaustive",
        "results/kapur/sa/",
        "results/kapur/vns/",
    ];
    for dir in dirs.iter() {
        fs::create_dir_all(dir).expect("Failed to create directory");
        for k in 2..=5 {
            let dir_path = format!("{}/k{k}/histogram", dir);
            fs::create_dir_all(&dir_path).expect("Failed to create directory");
        }
    }
    println!("meep");

    // Iterate over all files in the directory
    for entry in fs::read_dir(img_dir).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read directory entry");

        process_file(entry);
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Application error: {}", e);
        std::process::exit(1);
    }

    // Pause before exit
    println!("Results saved in the results folder where this program is located");
    println!("Run outputs saved to csv files where this program is located");
    println!("Press Enter to close...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}

fn process_file(entry: fs::DirEntry) {
    let img_path = entry.path();
    // Check if the entry is a file and has an image extension
    if img_path.is_file()
        && img_path
            .extension()
            .map_or(false, |ext| ext == "jpg" || ext == "png" || ext == "jpeg")
    {
        process_image(&img_path);
    }
}

fn process_image(img_path: &PathBuf) {
    // Read the image

    let img = image::open(&img_path).expect("Failed to open image");

    // Convert the image to grayscale (if it's not already)
    let gray_img = img.to_luma8();

    // Histogram exploration
    let exclude_zero = true;
    //explore_histgram(img_path, &gray_img, exclude_zero);

    let to_run: Vec<(
        &str,
        &str,
        fn(&str, &image::ImageBuffer<Luma<u8>, Vec<u8>>, usize) -> Vec<u8>,
    )> = vec![
        (
            "otsu",
            "sa",
            otsu::compute_otsu_thresholds_simulated_annealing,
        ),
        (
            "kapur",
            "sa",
            kapur::compute_kapur_thresholds_simulated_annealing,
        ),
        (
            "otsu",
            "vns",
            otsu::compute_otsu_thresholds_variable_neighborhood,
        ),
        (
            "kapur",
            "vns",
            kapur::compute_kapur_thresholds_variable_neighborhood,
        ),
        // (
        //     "kapur",
        //     "exhaustive",
        //     kapur::compute_exhaustive_kapur_thresholds,
        // ),
        // (
        //     "otsu",
        //     "exhaustive",
        //     otsu::compute_exhaustive_otsu_thresholds,
        // ),
    ];
    let file_stem = img_path.file_stem().unwrap().to_string_lossy();
    println!(
        "Processing image {:?}\nWhat will be run:
    \n\tOtsu, k=2,3,4,5, SA X30
    \n\tOtsu, k=2,3,4,5, VNS X30
    \n\tKapur, k=2,3,4,5, SA X30
    \n\tKapur, k=2,3,4,5, VNS X30
    \n A total of 480 runs will be performed
    \nResults will be saved in the results folder where this program is located
    \nPress Enter to continue...",
        file_stem
    );
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    for k in 2..=5 {
        for _ in 0..30 {
            do_metric_thresholding(&to_run, img_path, &gray_img, k, exclude_zero);
        }
    }
}

pub fn explore_histgram(img_path: &PathBuf, gray_img: &GrayImage, exclude_zero: bool) {
    // Read the image
    let file_stem = img_path.file_stem().unwrap().to_string_lossy();
    let output_path = format!("histograms/{}_histogram.png", file_stem);
    histogram_drawer::save_histogram(&file_stem, &gray_img, &output_path, exclude_zero);
}

pub fn do_metric_thresholding(
    to_run: &Vec<(
        &str,
        &str,
        fn(image_name: &str, gray_img: &GrayImage, k: usize) -> Vec<u8>,
    )>,
    img_path: &PathBuf,
    gray_img: &GrayImage,
    k: usize,
    exclude_zero: bool,
) {
    let file_stem = img_path.file_stem().unwrap().to_string_lossy();
    for func in to_run.iter() {
        do_metric_method(
            func.0,
            func.1,
            func.2,
            &file_stem,
            gray_img,
            k,
            exclude_zero,
        );
    }
}

pub fn do_metric_method(
    metric_name: &str,
    method_name: &str,
    compute_thresholds_fn: fn(&str, &GrayImage, usize) -> Vec<u8>,
    file_stem: &str,
    gray_img: &GrayImage,
    k: usize,
    exclude_zero: bool,
) {
    println!("{}: {}", metric_name, method_name.replace('_', " "));
    let thresholds = compute_thresholds_fn(&file_stem, gray_img, k);
    let base_path = format!(
        "results/{}/{}/k{}",
        metric_name,
        method_name.to_lowercase(),
        k
    );
    let histogram_path = format!(
        "{}/histogram/{}_k{}_histogram_{:?}.png",
        base_path, file_stem, k, thresholds
    );
    let segmented_path = format!("{}/{}_k{}_{:?}.png", base_path, file_stem, k, thresholds);
    draw_threshold_hist_and_save_image(
        &histogram_path,
        &segmented_path,
        file_stem,
        gray_img,
        &thresholds,
        exclude_zero,
        metric_name,
        method_name,
    );
}

fn draw_threshold_hist_and_save_image(
    histogram_path: &str,
    segmented_path: &str,
    file_stem: &str,
    gray_img: &GrayImage,
    thresholds: &Vec<u8>,
    exclude_zero: bool,
    method_name: &str,
    search_name: &str,
) {
    histogram_drawer::draw_histogram_with_thresholds(
        file_stem,
        method_name,
        search_name,
        gray_img,
        histogram_path,
        thresholds,
        exclude_zero,
    );
    let segmented_img = apply_thresholds(gray_img, thresholds);
    segmented_img
        .save(segmented_path)
        .expect("Failed to save image");
}

fn apply_thresholds(gray_img: &GrayImage, thresholds: &[u8]) -> image::RgbImage {
    let mut segmented_img = image::RgbImage::new(gray_img.width(), gray_img.height());
    let k = thresholds.len() + 1; // Number of classes

    // Define a set of colors for the segments
    let colors = [
        image::Rgb([0, 0, 255]),   // Blue
        image::Rgb([0, 255, 0]),   // Green
        image::Rgb([255, 0, 0]),   // Red
        image::Rgb([255, 255, 0]), // Yellow
        image::Rgb([255, 0, 255]), // Magenta
        image::Rgb([0, 255, 255]), // Cyan
    ];

    for (x, y, pixel) in gray_img.enumerate_pixels() {
        let intensity = pixel[0];
        let mut class = 0usize;
        for (i, &threshold) in thresholds.iter().enumerate() {
            if intensity > threshold {
                class = i + 1;
            } else {
                break;
            }
        }
        // Assign the color based on the class
        let color = colors[class % colors.len()];
        segmented_img.put_pixel(x, y, color);
    }
    segmented_img
}
