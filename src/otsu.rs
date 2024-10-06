use image::GrayImage;
use rand::prelude::*;

use crate::{file_writing, stats};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;


pub fn compute_exhaustive_otsu_thresholds(image_name:&str,gray_img: &GrayImage, k: usize) -> Vec<u8> {
    let start_time = Instant::now();
    if k < 2 {
        panic!("The number of classes 'k' must be at least 2.");
    }

    // Compute the histogram
    let mut histogram = [0u32; 256];
    let total_pixels = (gray_img.width() * gray_img.height()) as f64;

    for pixel in gray_img.pixels() {
        let intensity = pixel[0] as usize;
        histogram[intensity] += 1;
    }

    // Normalize histogram to get probabilities
    let prob: Vec<f64> = histogram.iter().map(|&count| count as f64 / total_pixels).collect();

    // Generate all possible combinations of thresholds
    let mut max_sigma = 0.0;
    let mut best_thresholds = vec![];

    let intensity_levels = 256;
    let thresholds_combinations = stats::combinations(1, intensity_levels - 1, k - 1);

    let pb = ProgressBar::new(thresholds_combinations.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.white/gray} {pos:>7}/{len:7} {msg} [{duration_precise}]",
        )
        .unwrap(),
    );

    for thresholds in thresholds_combinations {
        pb.inc(1);
        let sigma = stats::calculate_between_class_variance(&prob, &thresholds, intensity_levels);
        if sigma > max_sigma {
            max_sigma = sigma;
            best_thresholds = thresholds.clone();
        }
    }
    pb.finish_with_message("Done");

    let duration = start_time.elapsed();
   
    file_writing::writeln("otsu_exhaustive_times.csv", image_name, k, &best_thresholds, duration, max_sigma);

    println!("Optimal thresholds: {:?}", best_thresholds);

    // Convert thresholds to u8
    best_thresholds.iter().map(|&t| t as u8).collect()
}


pub fn compute_otsu_thresholds_simulated_annealing(image_name:&str,gray_img: &GrayImage, k: usize) -> Vec<u8> {
    let start_time = Instant::now();
    if k < 2 {
        panic!("The number of classes 'k' must be at least 2.");
    }

    // Compute the histogram
    let mut histogram = [0u32; 256];
    let total_pixels = (gray_img.width() * gray_img.height()) as f64;

    for pixel in gray_img.pixels() {
        let intensity = pixel[0] as usize;
        histogram[intensity] += 1;
    }

    // Normalize histogram to get probabilities
    let prob: Vec<f64> = histogram.iter().map(|&count| count as f64 / total_pixels).collect();

    // Initialize thresholds
    let intensity_levels = 256;
    let mut rng = StdRng::seed_from_u64(42);
    let mut thresholds: Vec<usize> = (0..k - 1).map(|_| rng.gen_range(1..255)).collect();
    thresholds.sort();

    // Compute initial between-class variance
    let mut max_sigma = stats::calculate_between_class_variance(&prob, &thresholds, intensity_levels);
    let mut best_thresholds = thresholds.clone();

    // Simulated annealing parameters
    let mut current_thresholds = thresholds;
    let mut current_sigma = max_sigma;

    let T0 = 100.0; // Initial temperature
    let alpha = 0.995; // Cooling rate
    let num_iterations = 100_000;//(((255 as u64).pow((k-1) as u32)) as f64 * 0.80).round() as u64;
    let mut T = T0;

    let pb = ProgressBar::new(num_iterations);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.white/gray} {pos:>7}/{len:7} {msg} [{duration_precise}]",
        )
        .unwrap(),
    );

    let mut no_improvement_count = 0;
    let max_no_improvement = 500; // Set a limit for iterations without improvement
    let  mut iter = 0;
    for _ in 0..num_iterations {
        if no_improvement_count >= max_no_improvement {
            break;
        }
        iter += 1;
        pb.inc(1);

        // Generate neighbor
        let mut neighbor_thresholds = current_thresholds.clone();

        // Randomly choose a threshold to modify
        let i = rng.gen_range(0..(k - 1));

        // Decide to increment or decrement
        let delta = if rng.gen_bool(0.5) { 1 } else { -1 };

        // Modify the threshold, ensuring valid range and order
        let mut new_value = neighbor_thresholds[i] as isize + delta;

        // Ensure new_value is within valid intensity range
        new_value = new_value.clamp(1, 254);

        // Ensure thresholds remain ordered
        if i > 0 && new_value <= neighbor_thresholds[i - 1] as isize {
            new_value = (neighbor_thresholds[i - 1] as isize + 1).clamp(1, 254);
        }
        if i < k - 2 && new_value >= neighbor_thresholds[i + 1] as isize {
            new_value = (neighbor_thresholds[i + 1] as isize - 1).clamp(1, 254);
        }

        // Update the threshold
        neighbor_thresholds[i] = new_value as usize;

        // Compute the between-class variance for the neighbor
        let neighbor_sigma =
            stats::calculate_between_class_variance(&prob, &neighbor_thresholds, intensity_levels);

        // Compute energy difference (we are maximizing sigma)
        let delta_e = neighbor_sigma - current_sigma;

        // Acceptance probability
        if delta_e >= 0.0 || rng.gen::<f64>() < f64::exp(delta_e / T) {
            // Accept neighbor
            current_thresholds = neighbor_thresholds.clone();
            current_sigma = neighbor_sigma;

            // Update best thresholds if necessary
            if neighbor_sigma > max_sigma {
                max_sigma = neighbor_sigma;
                best_thresholds = neighbor_thresholds;
                no_improvement_count = 0; // Reset no improvement counter
            } else {
                no_improvement_count += 1;
            }
        } else {
            no_improvement_count += 1;
        }

        // Update temperature
        T *= alpha;
    }
    pb.finish_with_message(format!("Done after {} iterations", iter));

    let duration = start_time.elapsed();
    file_writing::writeln("otsu_sa_times.csv", image_name, k, &best_thresholds, duration, max_sigma);

    println!("Optimal thresholds: {:?}", best_thresholds);

    // Convert thresholds to u8
    best_thresholds.iter().map(|&t| t as u8).collect()
}


pub fn compute_otsu_thresholds_variable_neighborhood(image_name: &str, gray_img: &GrayImage, k: usize) -> Vec<u8> {
    let start_time = Instant::now();
    if k < 2 {
        panic!("The number of classes 'k' must be at least 2.");
    }

    // Compute the histogram
    let mut histogram = [0u32; 256];
    let total_pixels = (gray_img.width() * gray_img.height()) as f64;

    for pixel in gray_img.pixels() {
        let intensity = pixel[0] as usize;
        histogram[intensity] += 1;
    }

    // Normalize histogram to get probabilities
    let prob: Vec<f64> = histogram.iter().map(|&count| count as f64 / total_pixels).collect();

    // Initialize thresholds to equally spaced values
    let intensity_levels = 256;
    let mut thresholds: Vec<usize> = (1..k).map(|i| i * 255 / k).collect();

    // Compute initial between-class variance
    let mut max_sigma = stats::calculate_between_class_variance(&prob, &thresholds, intensity_levels);
    let mut best_thresholds = thresholds.clone();

    // VNS parameters
    let mut rng = StdRng::seed_from_u64(42);
    let Kmax = 4;
    let num_iterations = 100_000;//(((255 as u64).pow((k-1) as u32)) as f64 * 0.80).round() as u64;
    let mut current_thresholds = thresholds.clone();
    let mut current_sigma = max_sigma;
    let mut k_neigh = 1;

    let pb = ProgressBar::new(num_iterations);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.white/gray} {pos:>7}/{len:7} {msg} [{duration_precise}]",
        )
        .unwrap(),
    );

    let mut iter = 0;

    let mut no_improvement_count = 0;
    let max_no_improvement = 500; // Set a limit for iterations without improvement

    while iter < num_iterations  {
        if  no_improvement_count >= max_no_improvement{
            break;
        }
        pb.inc(1);
        iter += 1;

        // Shaking
        let neighbor_thresholds = shaking(&mut rng, &current_thresholds, k_neigh, k);

        // Local Search
        let (local_best_thresholds, local_best_sigma) = local_search(&prob, neighbor_thresholds, intensity_levels);

        // Move or Not
        if local_best_sigma > current_sigma {
            current_thresholds = local_best_thresholds;
            current_sigma = local_best_sigma;
            if current_sigma > max_sigma {
                max_sigma = current_sigma;
                best_thresholds = current_thresholds.clone();
                no_improvement_count = 0; // Reset no improvement counter
            } else {
                no_improvement_count += 1;
            }
            k_neigh = 1;
        } else {
            no_improvement_count += 1;
            k_neigh += 1;
            if k_neigh > Kmax {
                k_neigh = 1;
            }
        }
    }
    pb.finish_with_message(format!("Done after {} iterations", iter));

    let duration = start_time.elapsed();
    file_writing::writeln("otsu_vns_times.csv", image_name, k, &best_thresholds, duration, max_sigma);

    println!("Optimal thresholds: {:?}", best_thresholds);

    // Convert thresholds to u8
    best_thresholds.iter().map(|&t| t as u8).collect()
}

fn shaking(rng: &mut StdRng, current_thresholds: &Vec<usize>, k_neigh: usize, k: usize) -> Vec<usize> {
    let mut neighbor_thresholds = current_thresholds.clone();

    match k_neigh {
        1 => {
            // Modify one threshold by ±1
            let i = rng.gen_range(0..k - 1);
            let delta = if rng.gen_bool(0.5) { 1 } else { -1 };
            let mut new_value = neighbor_thresholds[i] as isize + delta;
            new_value = new_value.clamp(1, 254);
            // Ensure thresholds remain ordered
            if i > 0 && new_value <= neighbor_thresholds[i - 1] as isize {
                new_value = neighbor_thresholds[i - 1] as isize + 1;
            }
            if i < k - 2 && new_value >= neighbor_thresholds[i + 1] as isize {
                new_value = neighbor_thresholds[i + 1] as isize - 1;
            }
            neighbor_thresholds[i] = new_value as usize;
        }
        2 => {
            // Modify one threshold by ±3
            let i = rng.gen_range(0..k - 1);
            let delta = if rng.gen_bool(0.5) { 3 } else { -3 };
            let mut new_value = neighbor_thresholds[i] as isize + delta;
            new_value = new_value.clamp(1, 254);
            // Ensure thresholds remain ordered
            if i > 0 && new_value <= neighbor_thresholds[i - 1] as isize {
                new_value = neighbor_thresholds[i - 1] as isize + 1;
            }
            if i < k - 2 && new_value >= neighbor_thresholds[i + 1] as isize {
                new_value = neighbor_thresholds[i + 1] as isize - 1;
            }
            neighbor_thresholds[i] = new_value as usize;
        }
        3 => {
            // Swap two thresholds
            if k - 1 >= 2 {
                let i = rng.gen_range(0..k - 1);
                let j = rng.gen_range(0..k - 1);
                if i != j {
                    neighbor_thresholds.swap(i, j);
                    neighbor_thresholds.sort();
                }
            }
        }
        4 => {
            // Replace one threshold with a random value
            let i = rng.gen_range(0..k - 1);
            let new_value = rng.gen_range(1..255);
            neighbor_thresholds[i] = new_value;
            neighbor_thresholds.sort();
        }
        _ => {}
    }

    neighbor_thresholds
}

fn local_search(prob: &Vec<f64>, initial_thresholds: Vec<usize>, intensity_levels: usize) -> (Vec<usize>, f64) {
    let mut current_thresholds = initial_thresholds.clone();
    let mut current_sigma = stats::calculate_between_class_variance(&prob, &current_thresholds, intensity_levels);
    let max_local_iterations = 100; // To prevent infinite loops
    let k = current_thresholds.len() + 1;

    for _ in 0..max_local_iterations {
        let mut improved = false;
        for i in 0..k -1 {
            for &delta in &[-1, 1] {
                let mut neighbor_thresholds = current_thresholds.clone();
                let mut new_value = neighbor_thresholds[i] as isize + delta;
                new_value = new_value.clamp(1, 254);
                // Ensure thresholds remain ordered
                if i > 0 && new_value <= neighbor_thresholds[i - 1] as isize {
                    continue;
                }
                if i < k - 2 && new_value >= neighbor_thresholds[i + 1] as isize {
                    continue;
                }
                neighbor_thresholds[i] = new_value as usize;
                let neighbor_sigma = stats::calculate_between_class_variance(&prob, &neighbor_thresholds, intensity_levels);
                if neighbor_sigma > current_sigma {
                    current_thresholds = neighbor_thresholds;
                    current_sigma = neighbor_sigma;
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
        if !improved {
            break;
        }
    }
    (current_thresholds, current_sigma)
}
