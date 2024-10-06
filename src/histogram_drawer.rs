use image::GrayImage;
use plotters::prelude::*;

pub fn save_histogram(
    image_name: &str,
    gray_img: &GrayImage,
    output_path: &str,
    exlude_zero: bool,
) {
    // Compute the histogram
    let mut histogram = [0u32; 256];

    for pixel in gray_img.pixels() {
        if exlude_zero && pixel[0] == 0 {
            continue;
        }
        let intensity = pixel[0] as usize;
        histogram[intensity] += 1;
    }

    // Plot the histogram
    let root = BitMapBackend::new(output_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_count = *histogram.iter().max().unwrap();
    let caption = if exlude_zero {
        format!("Histogram of {} (excluding zeros)", image_name)
    } else {
        format!("Histogram of {}", image_name)
    };
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0u32..255u32, 0u32..(max_count + max_count / 10))
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(histogram.iter().enumerate().map(|(x, y)| {
            let x0 = x as u32;
            let x1 = x0 + 1;
            let y0 = 0u32;
            let y1 = *y;
            Rectangle::new([(x0, y0), (x1, y1)], BLUE.mix(0.5).filled())
        }))
        .unwrap();

    // Ensure the output is saved
    root.present().unwrap();

    println!("Histogram saved to {}", output_path);
}


pub fn draw_histogram_with_thresholds(
    image_name: &str,
    method_name: &str,
    search_name:&str,
    gray_img: &GrayImage,
    output_path: &str,
    thresholds: &Vec<u8>,
    exclude_zero: bool
) {

    let mut histogram = [0u32; 256];

    for pixel in gray_img.pixels() {
        if exclude_zero && pixel[0] == 0 {
            continue;
        }
        let intensity = pixel[0] as usize;
        histogram[intensity] += 1;
    }

    let root = BitMapBackend::new(output_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let caption = format!("Histogram of {} segmented with {} using {}", image_name, method_name, search_name);
    let max_count = *histogram.iter().max().unwrap_or(&0);

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0u32..255u32, 0u32..(max_count + max_count / 10))
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(histogram.iter().enumerate().map(|(x, y)| {
            let x0 = x as u32;
            let x1 = x0 + 1;
            let y0 = 0u32;
            let y1 = *y;
            Rectangle::new([(x0, y0), (x1, y1)], BLUE.mix(0.5).filled())
        }))
        .unwrap();

    // Draw threshold lines
    for &threshold in thresholds {
        chart
            .draw_series(LineSeries::new(
                vec![(threshold as u32, 0), (threshold as u32, max_count)],
                &RED,
            ))
            .unwrap();
    }

    // Ensure the output is saved
    root.present().unwrap();

    //println!("Histogram saved to {}", output_path);
}


