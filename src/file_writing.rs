use std::{fs::OpenOptions, time::Duration};
use std::io::Write;
pub fn writeln(
    file_path: &str,
    image_name: &str,
    k: usize,
    best_thresholds: &Vec<usize>,
    duration: Duration,
    objective_value: f64,
) {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(file_path)
        .unwrap();
    writeln!(
        file,
        "{image_name},{k},{},[{}],{}",
        objective_value,
        best_thresholds
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";"),
        duration.as_secs_f64() * 1000.0,
    )
    .unwrap();
}
