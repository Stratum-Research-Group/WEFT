use eframe::egui;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize)]
struct LogEntry {
    #[serde(default)]
    step_index: u64,
    #[serde(default)]
    source: String,
    #[serde(rename = "type")]
    #[serde(default)]
    entry_type: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    created_at: String,
}

#[derive(Debug, Clone)]
struct ActiveTask {
    task_id: String,
    log_file: PathBuf,
    size_bytes: u64,
    recent_line: String,
}

struct WeftStudioApp {
    server_url: String,
    prompt: String,
    inference_output: String,
    transcript_path: PathBuf,
    tasks_dir: PathBuf,
    output_dir: PathBuf,
    logs: Vec<LogEntry>,
    active_tasks: Vec<ActiveTask>,
    selected_tab: usize,
    status_msg: String,
    auto_refresh: bool,
}

impl Default for WeftStudioApp {
    fn default() -> Self {
        let conversation_id = "32298e0e-3291-406d-a284-008552ab2cc9";
        let base_dir = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/akarta".to_string()))
            .join(".gemini/antigravity/brain")
            .join(conversation_id);

        let output_folder = PathBuf::from("/home/akarta/Projects/0.WiP/Weft/tools/weft_studio/output");
        std::fs::create_dir_all(&output_folder).ok();

        let mut app = Self {
            server_url: "http://127.0.0.1:8089".to_string(),
            prompt: "What is the status of the 1.04T MoE router?".to_string(),
            inference_output: "Ready to prompt local WEFT server...".to_string(),
            transcript_path: base_dir.join(".system_generated/logs/transcript.jsonl"),
            tasks_dir: base_dir.join(".system_generated/tasks"),
            output_dir: output_folder,
            logs: Vec::new(),
            active_tasks: Vec::new(),
            selected_tab: 0,
            status_msg: "Initialized".to_string(),
            auto_refresh: true,
        };
        app.reload_data();
        app
    }
}

impl WeftStudioApp {
    fn reload_data(&mut self) {
        // 1. Read last 50 chat messages from transcript.jsonl
        if self.transcript_path.exists() {
            if let Ok(file) = File::open(&self.transcript_path) {
                let reader = BufReader::new(file);
                let mut entries: Vec<LogEntry> = Vec::new();
                for line in reader.lines().flatten() {
                    if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
                        if entry.entry_type == "USER_INPUT"
                            || entry.entry_type == "PLANNER_RESPONSE"
                            || entry.entry_type == "SYSTEM_MESSAGE"
                        {
                            entries.push(entry);
                        }
                    }
                }
                if entries.len() > 60 {
                    self.logs = entries.split_off(entries.len() - 60);
                } else {
                    self.logs = entries;
                }
            }
        }

        // 2. Scan tasks directory
        self.active_tasks.clear();
        if self.tasks_dir.exists() {
            if let Ok(read_dir) = std::fs::read_dir(&self.tasks_dir) {
                for entry in read_dir.flatten() {
                    let path = entry.path();
                    if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("log") {
                        let task_id = path.file_stem().unwrap().to_string_lossy().to_string();
                        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                        let mut recent_line = String::new();
                        if let Ok(f) = File::open(&path) {
                            let r = BufReader::new(f);
                            if let Some(l) = r.lines().flatten().last() {
                                recent_line = l;
                            }
                        }
                        self.active_tasks.push(ActiveTask {
                            task_id,
                            log_file: path,
                            size_bytes: size,
                            recent_line,
                        });
                    }
                }
            }
            self.active_tasks.sort_by(|a, b| b.task_id.cmp(&a.task_id));
            if self.active_tasks.len() > 20 {
                self.active_tasks.truncate(20);
            }
        }
        self.status_msg = format!("Loaded {} messages, {} tasks", self.logs.len(), self.active_tasks.len());
    }

    fn query_model(&mut self) {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap();

        let payload = serde_json::json!({
            "model": "weft-subbit",
            "messages": [{"role": "user", "content": self.prompt}]
        });

        match client
            .post(format!("{}/v1/chat/completions", self.server_url))
            .json(&payload)
            .send()
        {
            Ok(resp) => {
                if let Ok(text) = resp.text() {
                    self.inference_output = text.clone();
                    // Save response to output folder
                    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
                    let out_path = self.output_dir.join(format!("response_{}.json", ts));
                    std::fs::write(&out_path, &text).ok();
                    self.status_msg = format!("Saved response to {}", out_path.display());
                } else {
                    self.inference_output = "Error reading response body".to_string();
                }
            }
            Err(e) => {
                self.inference_output = format!("Connection error to {}: {}", self.server_url, e);
            }
        }
    }
}

impl eframe::App for WeftStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.auto_refresh {
            ctx.request_repaint_after(Duration::from_millis(1500));
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("⚡ WEFT Frontier Studio & Antigravity Live Monitor");
                ui.separator();
                if ui.button("🔄 Refresh Data").clicked() {
                    self.reload_data();
                }
                ui.checkbox(&mut self.auto_refresh, "Auto-refresh (1.5s)");
                ui.separator();
                ui.label(format!("Status: {}", self.status_msg));
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.selected_tab, 0, "💬 Antigravity Chat Log");
                ui.selectable_value(&mut self.selected_tab, 1, "⚙ Active Tasks & Logs");
                ui.selectable_value(&mut self.selected_tab, 2, "🤖 Model Inference & Generator");
                ui.selectable_value(&mut self.selected_tab, 3, "📁 Output Artifacts Directory");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.selected_tab {
                0 => {
                    // Chat Log Viewer
                    ui.label(format!("Live Feed: {}", self.transcript_path.display()));
                    ui.separator();
                    egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                        for log in &self.logs {
                            let (bg_color, title) = match log.entry_type.as_str() {
                                "USER_INPUT" => (egui::Color32::from_rgb(30, 45, 60), "👤 USER REQUEST"),
                                "PLANNER_RESPONSE" => (egui::Color32::from_rgb(40, 50, 40), "🤖 ASSISTANT (Antigravity)"),
                                _ => (egui::Color32::from_rgb(45, 45, 45), "⚙ SYSTEM MESSAGE"),
                            };

                            egui::Frame::none()
                                .fill(bg_color)
                                .inner_margin(8.0)
                                .outer_margin(4.0)
                                .rounding(4.0)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.strong(title);
                                        ui.label(format!("Step #{} • {}", log.step_index, log.created_at));
                                    });
                                    if let Some(content) = &log.content {
                                        ui.monospace(content);
                                    }
                                });
                        }
                    });
                }
                1 => {
                    // Running & Recent Tasks
                    ui.label(format!("Monitoring Tasks in: {}", self.tasks_dir.display()));
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        egui::Grid::new("tasks_grid")
                            .striped(true)
                            .min_col_width(120.0)
                            .show(ui, |ui| {
                                ui.strong("Task ID");
                                ui.strong("Log Size");
                                ui.strong("Recent Output / Log Preview");
                                ui.end_row();

                                for task in &self.active_tasks {
                                    ui.label(&task.task_id);
                                    ui.label(format!("{:.1} KB", task.size_bytes as f64 / 1024.0));
                                    ui.monospace(&task.recent_line);
                                    ui.end_row();
                                }
                            });
                    });
                }
                2 => {
                    // Model Interaction
                    ui.horizontal(|ui| {
                        ui.label("Server Endpoint:");
                        ui.text_edit_singleline(&mut self.server_url);
                    });
                    ui.separator();
                    ui.label("Prompt Model:");
                    ui.text_edit_multiline(&mut self.prompt);
                    if ui.button("🚀 Send to Local WEFT Server").clicked() {
                        self.query_model();
                    }
                    ui.separator();
                    ui.label("Response Output:");
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.monospace(&self.inference_output);
                    });
                }
                3 => {
                    // Output Directory Browser
                    ui.label(format!("Items Folder: {}", self.output_dir.display()));
                    ui.separator();
                    if ui.button("📂 Open Folder in System File Manager").clicked() {
                        std::process::Command::new("xdg-open")
                            .arg(&self.output_dir)
                            .spawn()
                            .ok();
                    }
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if let Ok(entries) = std::fs::read_dir(&self.output_dir) {
                            for e in entries.flatten() {
                                ui.horizontal(|ui| {
                                    ui.label("📄");
                                    ui.label(e.file_name().to_string_lossy().to_string());
                                    if let Ok(meta) = e.metadata() {
                                        ui.label(format!("{:.1} KB", meta.len() as f64 / 1024.0));
                                    }
                                });
                            }
                        }
                    });
                }
                _ => {}
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 720.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("WEFT Frontier Studio"),
        ..Default::default()
    };
    eframe::run_native(
        "WEFT Frontier Studio",
        native_options,
        Box::new(|_cc| Ok(Box::new(WeftStudioApp::default()))),
    )
}
