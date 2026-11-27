use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkRecord {
    #[serde(default)]
    pub latency_ns: f64,
    #[serde(default)]
    pub throughput_mtrans_per_sec: f64,
    #[serde(default)]
    pub memory_bandwidth_gb_per_sec: f64,
    #[serde(default)]
    pub flop_efficiency_percent: f64,
    #[serde(default)]
    pub status: String,
}

#[derive(Debug, Clone, Default)]
pub struct HardwareTelemetry {
    pub benchmarks: HashMap<String, BenchmarkRecord>,
    pub total_benchmarks: usize,
    pub passing_benchmarks: usize,
    pub avg_throughput_gtrans: f64,
}

impl HardwareTelemetry {
    pub fn load_from_default_path() -> Self {
        let path = Path::new("build/weft_empirical_benchmarks.json");
        if !path.exists() {
            return Self::default();
        }

        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Self::default(),
        };

        let mut content = String::new();
        if file.read_to_string(&mut content).is_err() {
            return Self::default();
        }

        let benchmarks: HashMap<String, BenchmarkRecord> = match serde_json::from_str(&content) {
            Ok(b) => b,
            Err(_) => return Self::default(),
        };

        let total = benchmarks.len();
        let passing = benchmarks.values().filter(|b| b.status == "PASS" || b.throughput_mtrans_per_sec > 0.0).count();
        let total_mtrans: f64 = benchmarks.values().map(|b| b.throughput_mtrans_per_sec).sum();
        let avg_throughput_gtrans = if total > 0 { (total_mtrans / total as f64) / 1000.0 } else { 0.0 };

        Self {
            benchmarks,
            total_benchmarks: total,
            passing_benchmarks: passing,
            avg_throughput_gtrans,
        }
    }
}
