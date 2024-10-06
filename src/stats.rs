// Function to generate all combinations of thresholds
pub fn combinations(start: usize, end: usize, k: usize) -> Vec<Vec<usize>> {
    fn combine(start: usize, end: usize, k: usize, prefix: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if k == 0 {
            result.push(prefix.clone());
            return;
        }
        for i in start..=end - k + 1 {
            prefix.push(i);
            combine(i + 1, end, k - 1, prefix, result);
            prefix.pop();
        }
    }

    let mut result = vec![];
    combine(start, end, k, &mut vec![], &mut result);
    result
}

// Function to calculate the between-class variance for given thresholds
pub fn calculate_between_class_variance(prob: &[f64], thresholds: &[usize], intensity_levels: usize) -> f64 {
    let mut class_prob = vec![];
    let mut class_mean = vec![];
    let mut start = 0;
    let mut total_mean = 0.0;

    for (i, &p) in prob.iter().enumerate() {
        total_mean += i as f64 * p;
    }

    for &threshold in thresholds.iter().chain(std::iter::once(&intensity_levels)) {
        let mut sum_prob = 0.0;
        let mut sum_mean = 0.0;
        for i in start..threshold {
            sum_prob += prob[i];
            sum_mean += i as f64 * prob[i];
        }
        if sum_prob > 0.0 {
            class_prob.push(sum_prob);
            class_mean.push(sum_mean / sum_prob);
        } else {
            class_prob.push(0.0);
            class_mean.push(0.0);
        }
        start = threshold;
    }

    let mut sigma_between = 0.0;
    for (i, &p) in class_prob.iter().enumerate() {
        sigma_between += p * (class_mean[i] - total_mean).powi(2);
    }

    sigma_between
}