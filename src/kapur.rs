use std::{fs::OpenOptions, time::Instant};

use image::GrayImage;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use std::io::Write;
use crate::{file_writing, stats};

pub fn compute_exhaustive_kapur_thresholds(image_name:&str,gray_img: &GrayImage, k: usize) -> Vec<u8> {
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
    let intensity_levels = 256;
    let thresholds_combinations = stats::combinations(1, intensity_levels - 1, k - 1);

    let pb = ProgressBar::new(thresholds_combinations.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} [{duration_precise}]",
        )
        .unwrap(),
    );

    let mut max_entropy = f64::MIN;
    let mut best_thresholds = vec![];

    for thresholds in thresholds_combinations {
        pb.inc(1);
        let entropy = calculate_total_entropy(&prob, &thresholds, intensity_levels);
        if entropy > max_entropy {
            max_entropy = entropy;
            best_thresholds = thresholds.clone();
        }
    }
    pb.finish_with_message("Done");
    let duration = start_time.elapsed();
    file_writing::writeln("kapur_exhaustive_times.csv", image_name, k, &best_thresholds, duration, max_entropy);

    println!("Optimal thresholds (Kapur's method): {:?}", best_thresholds);

    // Convert thresholds to u8
    best_thresholds.iter().map(|&t| t as u8).collect()
}



pub fn compute_kapur_thresholds_simulated_annealing(image_name: &str, gray_img: &GrayImage, k: usize) -> Vec<u8> {
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

    // Simulated annealing parameters
    let intensity_levels = 256;
    let mut rng = StdRng::seed_from_u64(42);

    // Initial thresholds (evenly spaced)
    let mut current_thresholds: Vec<usize> = (1..k).map(|i| i * intensity_levels / k).collect();
    let mut best_thresholds = current_thresholds.clone();
    let mut current_entropy = calculate_total_entropy(&prob, &current_thresholds, intensity_levels);
    let mut best_entropy = current_entropy;

    let max_iterations = 100_000;//(((255 as u64).pow((k-1) as u32)) as f64 * 0.80).round() as u64;
    let mut temp = 100.0; // Initial temperature
    let min_temp = 1e-3;
    let alpha = 0.99; // Cooling rate

    let pb = ProgressBar::new(max_iterations);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} [{duration_precise}]",
        )
        .unwrap(),
    );

    let mut no_improvement_count = 0;
    let max_no_improvement = 500; // Set number of iterations with no improvement

    for _ in 0..max_iterations {
        pb.inc(1);
        if temp < min_temp || no_improvement_count >= max_no_improvement {
            break;
        }

        // Generate neighbor thresholds
        let mut new_thresholds = current_thresholds.clone();

        // Randomly choose one threshold to modify
        let idx = rng.gen_range(0..k - 1);

        // Modify the threshold by a small random amount
        let delta = rng.gen_range(-1..=1); // Can be -1, 0, or 1

        new_thresholds[idx] = (new_thresholds[idx] as isize + delta)
            .clamp(1, intensity_levels as isize - 1) as usize;

        // Ensure thresholds are in order
        new_thresholds.sort_unstable();

        // Compute entropy for new thresholds
        let new_entropy = calculate_total_entropy(&prob, &new_thresholds, intensity_levels);

        // Decide whether to accept the new thresholds
        let delta_entropy = new_entropy - current_entropy;
        if delta_entropy > 0.0 || rng.gen_bool((delta_entropy / temp).exp().min(1.0)) {
            // Accept new thresholds
            current_thresholds = new_thresholds.clone();
            current_entropy = new_entropy;

            // Update best if necessary
            if new_entropy > best_entropy {
                best_entropy = new_entropy;
                best_thresholds = new_thresholds;
                no_improvement_count = 0; // Reset no improvement count
            }
        } else {
            no_improvement_count += 1;
        }

        // Cool down the temperature
        temp *= alpha;
    }
    pb.finish_with_message("Done");
    let duration = start_time.elapsed();
    file_writing::writeln("kapur_sa_times.csv", image_name, k, &best_thresholds, duration, best_entropy);

    println!("Optimal thresholds (Kapur's method with SA): {:?}", best_thresholds);

    // Convert thresholds to u8
    best_thresholds.iter().map(|&t| t as u8).collect()
}


// Function to calculate the total entropy for given thresholds
fn calculate_total_entropy(prob: &[f64], thresholds: &[usize], intensity_levels: usize) -> f64 {
    let mut total_entropy = 0.0;
    let mut start = 0;

    for &threshold in thresholds.iter().chain(std::iter::once(&intensity_levels)) {
        // Calculate the class probability (sum of probabilities within the class)
        let class_prob: f64 = prob[start..threshold].iter().sum();

        if class_prob > 0.0 {
            let mut entropy = 0.0;
            // Compute the entropy using normalized probabilities within the class
            for i in start..threshold {
                let p = prob[i] / class_prob; // Normalize the probability
                if p > 0.0 {
                    entropy -= p * p.ln(); // Use natural logarithm
                }
            }
            total_entropy += entropy;
        }
        start = threshold;
    }
    total_entropy
}

pub fn compute_kapur_thresholds_variable_neighborhood(image_name: &str, gray_img: &GrayImage, k: usize) -> Vec<u8> {
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

    // Compute initial total entropy
    let mut max_entropy = calculate_total_entropy(&prob, &thresholds, intensity_levels);
    let mut best_thresholds = thresholds.clone();

    // VNS parameters
    let mut rng = StdRng::seed_from_u64(42);
    let Kmax = 4;
    let num_iterations = 100_000;//(((255 as u64).pow((k-1) as u32)) as f64 * 0.80).round() as u64;
    let mut current_thresholds = thresholds.clone();
    let mut current_entropy = max_entropy;
    let mut k_neigh = 1;

    let pb = ProgressBar::new(num_iterations);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} [{duration_precise}]",
        )
        .unwrap(),
    );

    let mut iter = 0;

    let mut no_improvement_count = 0;
    let max_no_improvement = 500; // Set number of iterations with no improvement

    while iter < num_iterations && no_improvement_count < max_no_improvement {
        pb.inc(1);
        iter += 1;

        // Shaking
        let neighbor_thresholds = shaking(&mut rng, &current_thresholds, k_neigh, k);

        // Local Search
        let (local_best_thresholds, local_best_entropy) =
            local_search(&prob, neighbor_thresholds, intensity_levels);

        // Move or Not
        if local_best_entropy > current_entropy {
            current_thresholds = local_best_thresholds;
            current_entropy = local_best_entropy;
            if current_entropy > max_entropy {
                max_entropy = current_entropy;
                best_thresholds = current_thresholds.clone();
            }
            k_neigh = 1;
            no_improvement_count = 0; // Reset no improvement count
        } else {
            k_neigh += 1;
            no_improvement_count += 1;
            if k_neigh > Kmax {
                k_neigh = 1;
            }
        }
    }
    pb.finish_with_message(format!("Done after {} iterations", iter));

    let duration = start_time.elapsed();
    file_writing::writeln("kapur_vns_times.csv", image_name, k, &best_thresholds, duration, max_entropy);
    println!("Optimal thresholds (Kapur's method with VNS): {:?}", best_thresholds);

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

fn local_search(
    prob: &Vec<f64>,
    initial_thresholds: Vec<usize>,
    intensity_levels: usize,
) -> (Vec<usize>, f64) {
    let mut current_thresholds = initial_thresholds.clone();
    let mut current_entropy = calculate_total_entropy(&prob, &current_thresholds, intensity_levels);
    let max_local_iterations = 100; // To prevent infinite loops
    let k = current_thresholds.len() + 1;

    for _ in 0..max_local_iterations {
        let mut improved = false;
        for i in 0..k - 1 {
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
                let neighbor_entropy =
                    calculate_total_entropy(&prob, &neighbor_thresholds, intensity_levels);
                if neighbor_entropy > current_entropy {
                    current_thresholds = neighbor_thresholds;
                    current_entropy = neighbor_entropy;
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
    (current_thresholds, current_entropy)
}

