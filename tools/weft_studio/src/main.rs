use eframe::egui;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
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
    content: Option<String>,
    #[serde(default)]
    created_at: String,
}

#[derive(Debug, Clone)]
struct ActiveTask {
    task_id: String,
    size_bytes: u64,
    recent_line: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
    timestamp: String,
    model_name: String,
    domain: String,
}

#[derive(Debug, Clone, PartialEq)]
enum BackendType {
    Ollama(&'static str),
    WeftLivePort(&'static str),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CouncilResponse {
    model_name: String,
    domain: String,
    text: String,
    status: String,
}

struct WeftStudioApp {
    // Interactive chat state
    selected_model_idx: usize,
    selected_domain: String,
    chat_input: String,
    imf_chat_history: Vec<ChatMessage>,
    weft_chat_history: Vec<ChatMessage>,
    council_history: Vec<(String, Vec<CouncilResponse>)>, // (User prompt, council answers)

    // Antigravity logs state
    weft_transcript_path: PathBuf,
    imf_transcript_path: PathBuf,
    weft_logs: Vec<LogEntry>,
    imf_logs: Vec<LogEntry>,
    weft_tasks: Vec<ActiveTask>,
    imf_tasks: Vec<ActiveTask>,

    output_dir: PathBuf,
    selected_tab: usize, // 0: Dual Chat (IMF | WEFT), 1: 🏛 Council of Models, 2: 📜 Antigravity Dual Logs, 3: ⚙ Active Background Tasks, 4: 📁 Output Artifacts
    status_msg: String,
    auto_refresh: bool,
    is_generating: bool,

    // Channels for background generation
    tx: Sender<(String, ChatMessage, bool)>, // (target_column: "weft"|"imf", msg, is_done)
    rx: Receiver<(String, ChatMessage, bool)>,
    council_tx: Sender<(usize, CouncilResponse)>,
    council_rx: Receiver<(usize, CouncilResponse)>,
}

const MODELS: &[(&str, &str, BackendType)] = &[
    ("Qwen 2.5 (14B High-Speed)", "qwen2.5:14b", BackendType::Ollama("qwen2.5:14b")),
    ("Qwen 3.8 (27B Frontier Reasoning)", "qwen3.8-27b:latest", BackendType::Ollama("qwen3.8-27b:latest")),
    ("Qwen 2.5-VL (7B Vision-Language)", "qwen2.5vl:7b", BackendType::Ollama("qwen2.5vl:7b")),
    ("WEFT Native Sub-Bit (Qwen2.5-Coder 0.5B GPU)", "weft-qwen-coder-0.5b", BackendType::WeftLivePort("http://127.0.0.1:8089")),
];

const EXPERT_DOMAINS: &[&str] = &[
    "General Intelligence & Synthesis",
    "Code Generation & System Programming",
    "Analytic Number Theory & RH",
    "Algebraic Geometry & Diophantine",
    "Formal Lean 4 Verification",
    "Blackwell Triton Kernel Acceleration",
];

impl Default for WeftStudioApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let (council_tx, council_rx) = channel();

        let weft_conv_id = "32298e0e-3291-406d-a284-008552ab2cc9";
        let imf_conv_id = "709ff001-bb91-4c96-a70e-71e9c075fc42";

        let brain_base = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/akarta".to_string()))
            .join(".gemini/antigravity/brain");

        let output_folder = PathBuf::from("/home/akarta/Projects/0.WiP/Weft/tools/weft_studio/output");
        std::fs::create_dir_all(&output_folder).ok();

        let now = chrono::Local::now().format("%H:%M:%S").to_string();

        let mut app = Self {
            selected_model_idx: 0,
            selected_domain: EXPERT_DOMAINS[0].to_string(),
            chat_input: String::new(),
            imf_chat_history: vec![ChatMessage {
                role: "assistant".to_string(),
                content: "IMF Autonomous Engine online. Ready for Hyperbolic Geometry, MegaGym 32B, and Neural-Symbolic distillation.".to_string(),
                timestamp: now.clone(),
                model_name: "Qwen 2.5 (14B)".to_string(),
                domain: "IMF Kernel".to_string(),
            }],
            weft_chat_history: vec![ChatMessage {
                role: "assistant".to_string(),
                content: "WEFT Sub-Bit Engine online. Connected to local GPU Blackwell inference and real safetensors factorization.".to_string(),
                timestamp: now,
                model_name: "WEFT Native".to_string(),
                domain: "WEFT Core".to_string(),
            }],
            council_history: Vec::new(),

            weft_transcript_path: brain_base.join(weft_conv_id).join(".system_generated/logs/transcript.jsonl"),
            imf_transcript_path: brain_base.join(imf_conv_id).join(".system_generated/logs/transcript.jsonl"),
            weft_logs: Vec::new(),
            imf_logs: Vec::new(),
            weft_tasks: Vec::new(),
            imf_tasks: Vec::new(),

            output_dir: output_folder,
            selected_tab: 0,
            status_msg: "Fleet Ready (All models local on NVMe/GPU)".to_string(),
            auto_refresh: true,
            is_generating: false,
            tx,
            rx,
            council_tx,
            council_rx,
        };

        app.reload_logs();
        app
    }
}

impl WeftStudioApp {
    fn reload_logs(&mut self) {
        // Read WEFT transcript
        self.weft_logs = Self::read_transcript_tail(&self.weft_transcript_path);
        // Read IMF transcript
        self.imf_logs = Self::read_transcript_tail(&self.imf_transcript_path);

        // Read background tasks
        let brain_base = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/akarta".to_string()))
            .join(".gemini/antigravity/brain");
        self.weft_tasks = Self::read_active_tasks(&brain_base.join("32298e0e-3291-406d-a284-008552ab2cc9/.system_generated/tasks"));
        self.imf_tasks = Self::read_active_tasks(&brain_base.join("709ff001-bb91-4c96-a70e-71e9c075fc42/.system_generated/tasks"));
    }

    fn read_transcript_tail(path: &PathBuf) -> Vec<LogEntry> {
        let mut entries = Vec::new();
        if let Ok(mut file) = File::open(path) {
            let meta = file.metadata().ok();
            let len = meta.map(|m| m.len()).unwrap_or(0);
            let seek_pos = if len > 250_000 { len - 250_000 } else { 0 };
            let _ = file.seek(SeekFrom::Start(seek_pos));
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
                    entries.push(entry);
                }
            }
        }
        entries
    }

    fn read_active_tasks(tasks_dir: &PathBuf) -> Vec<ActiveTask> {
        let mut tasks = Vec::new();
        if let Ok(dir_entries) = std::fs::read_dir(tasks_dir) {
            for entry in dir_entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("log") {
                    let task_id = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    let mut recent_line = String::new();
                    if let Ok(file) = File::open(&path) {
                        let r = BufReader::new(file);
                        if let Some(Ok(last)) = r.lines().last() {
                            recent_line = last;
                        }
                    }
                    tasks.push(ActiveTask {
                        task_id,
                        size_bytes,
                        recent_line,
                    });
                }
            }
        }
        tasks
    }

    fn query_model(backend: BackendType, system_prompt: &str, user_prompt: &str) -> Result<String, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| e.to_string())?;

        match backend {
            BackendType::Ollama(model_tag) => {
                let payload = serde_json::json!({
                    "model": model_tag,
                    "prompt": format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", system_prompt, user_prompt),
                    "stream": false,
                    "options": {
                        "temperature": 0.6,
                        "num_predict": 512
                    }
                });

                let res = client.post("http://127.0.0.1:11434/api/generate")
                    .json(&payload)
                    .send()
                    .map_err(|e| format!("Ollama connection error: {}", e))?;

                let val = res.json::<serde_json::Value>()
                    .map_err(|e| format!("JSON parse error: {}", e))?;

                if let Some(r) = val["response"].as_str() {
                    // Filter out think tags if present
                    let clean = if let Some(idx) = r.find("</think>") {
                        &r[idx + 8..]
                    } else {
                        r
                    };
                    Ok(clean.trim().to_string())
                } else {
                    Err("No response field in Ollama output".to_string())
                }
            }
            BackendType::WeftLivePort(url) => {
                let payload = serde_json::json!({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 512
                });

                let res = client.post(format!("{}/v1/chat/completions", url))
                    .json(&payload)
                    .send()
                    .map_err(|e| format!("WEFT Live Server error: {}", e))?;

                let val = res.json::<serde_json::Value>()
                    .map_err(|e| format!("JSON parse error: {}", e))?;

                if let Some(content) = val["choices"][0]["message"]["content"].as_str() {
                    Ok(content.trim().to_string())
                } else {
                    Err("Unexpected live server output format".to_string())
                }
            }
        }
    }

    fn dispatch_chat(&mut self, target_channel: &str) {
        if self.chat_input.trim().is_empty() || self.is_generating {
            return;
        }

        let user_text = self.chat_input.trim().to_string();
        self.chat_input.clear();
        let now = chrono::Local::now().format("%H:%M:%S").to_string();
        let (display_name, _, backend) = MODELS[self.selected_model_idx].clone();
        let domain = self.selected_domain.clone();

        let user_msg = ChatMessage {
            role: "user".to_string(),
            content: user_text.clone(),
            timestamp: now,
            model_name: display_name.to_string(),
            domain: domain.clone(),
        };

        if target_channel == "imf" {
            self.imf_chat_history.push(user_msg);
        } else {
            self.weft_chat_history.push(user_msg);
        }

        self.is_generating = true;
        self.status_msg = format!("Generating response from {}...", display_name);

        let tx = self.tx.clone();
        let target = target_channel.to_string();
        let sys_prompt = format!("You are an expert specialist in {}. Respond with rigorous, helpful technical precision.", domain);

        thread::spawn(move || {
            let reply_text = match Self::query_model(backend, &sys_prompt, &user_text) {
                Ok(t) => t,
                Err(e) => format!("Error connecting to local model: {}", e),
            };

            let reply_msg = ChatMessage {
                role: "assistant".to_string(),
                content: reply_text,
                timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                model_name: display_name.to_string(),
                domain,
            };

            let _ = tx.send((target, reply_msg, true));
        });
    }

    fn dispatch_council(&mut self) {
        if self.chat_input.trim().is_empty() || self.is_generating {
            return;
        }

        let user_text = self.chat_input.trim().to_string();
        self.chat_input.clear();

        let council_idx = self.council_history.len();
        let initial_responses: Vec<CouncilResponse> = MODELS.iter().map(|(name, _, _)| {
            CouncilResponse {
                model_name: name.to_string(),
                domain: self.selected_domain.clone(),
                text: "Querying model...".to_string(),
                status: "Pending".to_string(),
            }
        }).collect();

        self.council_history.push((user_text.clone(), initial_responses));
        self.is_generating = true;
        self.status_msg = "Convening Model Council across all local engines...".to_string();

        let domain = self.selected_domain.clone();

        for (i, (name, _, backend)) in MODELS.iter().enumerate() {
            let tx = self.council_tx.clone();
            let b = backend.clone();
            let m_name = name.to_string();
            let d_name = domain.clone();
            let prompt = user_text.clone();
            let sys_prompt = format!("You are member '{}' of the Frontier AI Council specializing in {}. Provide your highest quality analysis.", m_name, d_name);

            thread::spawn(move || {
                let ans = match Self::query_model(b, &sys_prompt, &prompt) {
                    Ok(t) => t,
                    Err(e) => format!("Error: {}", e),
                };

                let resp = CouncilResponse {
                    model_name: m_name,
                    domain: d_name,
                    text: ans,
                    status: "Completed".to_string(),
                };

                let _ = tx.send((council_idx, resp));
            });
        }
    }

    fn check_messages(&mut self) {
        while let Ok((target, msg, _done)) = self.rx.try_recv() {
            if target == "imf" {
                self.imf_chat_history.push(msg);
            } else {
                self.weft_chat_history.push(msg);
            }
            self.is_generating = false;
            self.status_msg = "Response received".to_string();

            // Save conversation snapshot
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            let out_path = self.output_dir.join(format!("{}_chat_{}.json", target, ts));
            let data = if target == "imf" { &self.imf_chat_history } else { &self.weft_chat_history };
            if let Ok(file_data) = serde_json::to_string_pretty(data) {
                let _ = std::fs::write(&out_path, file_data);
            }
        }

        while let Ok((c_idx, resp)) = self.council_rx.try_recv() {
            if c_idx < self.council_history.len() {
                if let Some(slot) = self.council_history[c_idx].1.iter_mut().find(|r| r.model_name == resp.model_name) {
                    *slot = resp;
                }
            }
            self.is_generating = false;
            self.status_msg = "Council deliberations updated".to_string();

            // Save council snapshot
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            let out_path = self.output_dir.join(format!("council_deliberation_{}.json", ts));
            if let Ok(file_data) = serde_json::to_string_pretty(&self.council_history) {
                let _ = std::fs::write(&out_path, file_data);
            }
        }
    }
}

impl eframe::App for WeftStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_messages();

        if self.auto_refresh {
            ctx.request_repaint_after(Duration::from_millis(1000));
        }

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("⚡ WEFT & IMF Studio | Local Frontier AI Cockpit");
                ui.separator();
                if ui.button("🔄 Sync Transcripts & Tasks").clicked() {
                    self.reload_logs();
                }
                ui.checkbox(&mut self.auto_refresh, "Auto-Poll (1s)");
                ui.separator();
                ui.label(format!("Engine Status: {}", self.status_msg));
            });

            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.selected_tab, 0, "💬 Split Chat (WEFT | IMF)");
                ui.selectable_value(&mut self.selected_tab, 1, "🏛 Council of Models (Deliberate)");
                ui.selectable_value(&mut self.selected_tab, 2, "📜 Antigravity Dual Logs (WEFT | IMF)");
                ui.selectable_value(&mut self.selected_tab, 3, "⚙ Active Background Tasks");
                ui.selectable_value(&mut self.selected_tab, 4, "📁 Output Artifacts Browser");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.selected_tab {
                0 => {
                    // Split Chat (WEFT on left, IMF on right)
                    ui.horizontal(|ui| {
                        ui.label("🎯 Active Model:");
                        egui::ComboBox::from_id_salt("model_pick")
                            .selected_text(MODELS[self.selected_model_idx].0)
                            .show_ui(ui, |ui| {
                                for (i, (name, _, _)) in MODELS.iter().enumerate() {
                                    ui.selectable_value(&mut self.selected_model_idx, i, *name);
                                }
                            });

                        ui.separator();
                        ui.label("🔬 Expert Domain:");
                        egui::ComboBox::from_id_salt("domain_pick")
                            .selected_text(&self.selected_domain)
                            .show_ui(ui, |ui| {
                                for d in EXPERT_DOMAINS {
                                    ui.selectable_value(&mut self.selected_domain, d.to_string(), *d);
                                }
                            });
                        ui.label("🔒 100% Internal Local Physical Weights");
                    });
                    ui.separator();

                    let available_height = ui.available_height() - 50.0;
                    let half_width = (ui.available_width() - 20.0) / 2.0;

                    ui.horizontal(|ui| {
                        // WEFT Column
                        ui.allocate_ui([half_width, available_height].into(), |ui| {
                            ui.heading("🧵 WEFT Chat Column");
                            ui.label("Sub-bit quantization, 1.04T MoE, Blackwell kernels");
                            ui.separator();
                            egui::ScrollArea::vertical()
                                .id_salt("weft_chat_scroll")
                                .max_height(available_height - 40.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    for msg in &self.weft_chat_history {
                                        let bg = if msg.role == "user" {
                                            egui::Color32::from_rgb(25, 45, 65)
                                        } else {
                                            egui::Color32::from_rgb(25, 55, 35)
                                        };
                                        egui::Frame::none().fill(bg).inner_margin(8.0).rounding(6.0).show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.strong(if msg.role == "user" { "👤 You" } else { &msg.model_name });
                                                ui.label(format!("• {}", msg.timestamp));
                                            });
                                            ui.label(&msg.content);
                                        });
                                        ui.add_space(4.0);
                                    }
                                });
                        });

                        ui.separator();

                        // IMF Column
                        ui.allocate_ui([half_width, available_height].into(), |ui| {
                            ui.heading("🌐 IMF Chat Column");
                            ui.label("Hyperbolic Gyrovector, MegaGym 32B, Lean proofs");
                            ui.separator();
                            egui::ScrollArea::vertical()
                                .id_salt("imf_chat_scroll")
                                .max_height(available_height - 40.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    for msg in &self.imf_chat_history {
                                        let bg = if msg.role == "user" {
                                            egui::Color32::from_rgb(35, 35, 65)
                                        } else {
                                            egui::Color32::from_rgb(50, 40, 25)
                                        };
                                        egui::Frame::none().fill(bg).inner_margin(8.0).rounding(6.0).show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.strong(if msg.role == "user" { "👤 You" } else { &msg.model_name });
                                                ui.label(format!("• {}", msg.timestamp));
                                            });
                                            ui.label(&msg.content);
                                        });
                                        ui.add_space(4.0);
                                    }
                                });
                        });
                    });

                    ui.separator();
                    ui.horizontal(|ui| {
                        let text_edit = ui.add_sized(
                            [ui.available_width() - 220.0, 32.0],
                            egui::TextEdit::singleline(&mut self.chat_input).hint_text("Type a prompt... send to WEFT or IMF or Council"),
                        );

                        if ui.button("🧵 Send to WEFT").clicked() || (text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                            self.dispatch_chat("weft");
                        }
                        if ui.button("🌐 Send to IMF").clicked() {
                            self.dispatch_chat("imf");
                        }
                    });
                }
                1 => {
                    // Council of Models Tab (All models deliberate on the same prompt simultaneously)
                    ui.horizontal(|ui| {
                        ui.heading("🏛 The Frontier AI Council");
                        ui.label("| Concurrently queries all internal models & displays their independent views");
                        ui.separator();
                        ui.label("Domain:");
                        egui::ComboBox::from_id_salt("council_domain_pick")
                            .selected_text(&self.selected_domain)
                            .show_ui(ui, |ui| {
                                for d in EXPERT_DOMAINS {
                                    ui.selectable_value(&mut self.selected_domain, d.to_string(), *d);
                                }
                            });
                    });
                    ui.separator();

                    let available_height = ui.available_height() - 50.0;
                    egui::ScrollArea::vertical()
                        .id_salt("council_scroll")
                        .max_height(available_height)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            if self.council_history.is_empty() {
                                ui.label("No council sessions yet. Enter a question below and click '🏛 Convene Council' to ask all models simultaneously.");
                            }

                            for (query, responses) in &self.council_history {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(20, 30, 45))
                                    .inner_margin(10.0)
                                    .rounding(6.0)
                                    .show(ui, |ui| {
                                        ui.strong(format!("📋 Council Query: \"{}\"", query));
                                    });
                                ui.add_space(6.0);

                                for resp in responses {
                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(30, 35, 40))
                                        .inner_margin(8.0)
                                        .rounding(4.0)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.strong(format!("👑 {}", resp.model_name));
                                                ui.label(format!("• [{}] • Status: {}", resp.domain, resp.status));
                                            });
                                            ui.separator();
                                            ui.label(&resp.text);
                                        });
                                    ui.add_space(4.0);
                                }
                                ui.separator();
                            }
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        let text_edit = ui.add_sized(
                            [ui.available_width() - 180.0, 32.0],
                            egui::TextEdit::singleline(&mut self.chat_input).hint_text("Ask a question to the entire Model Council..."),
                        );

                        if ui.button("🏛 Convene Council").clicked() || (text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                            self.dispatch_council();
                        }
                    });
                }
                2 => {
                    // Dual Antigravity Logs
                    ui.horizontal(|ui| {
                        ui.heading("📜 Antigravity Live Transcripts (WEFT vs IMF Side-by-Side)");
                    });
                    ui.separator();

                    let available_height = ui.available_height();
                    let half_width = (ui.available_width() - 20.0) / 2.0;

                    ui.horizontal(|ui| {
                        // WEFT Transcript
                        ui.allocate_ui([half_width, available_height].into(), |ui| {
                            ui.strong("🧵 WEFT Session Log (Current)");
                            egui::ScrollArea::vertical().id_salt("weft_log_scroll").stick_to_bottom(true).show(ui, |ui| {
                                for log in &self.weft_logs {
                                    let bg = match log.entry_type.as_str() {
                                        "USER_INPUT" => egui::Color32::from_rgb(25, 40, 55),
                                        "PLANNER_RESPONSE" => egui::Color32::from_rgb(30, 45, 30),
                                        _ => egui::Color32::from_rgb(35, 35, 35),
                                    };
                                    egui::Frame::none().fill(bg).inner_margin(6.0).rounding(4.0).show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.strong(&log.entry_type);
                                            ui.label(format!("Step #{} • {}", log.step_index, log.created_at));
                                        });
                                        if let Some(content) = &log.content {
                                            ui.monospace(content);
                                        }
                                    });
                                    ui.add_space(3.0);
                                }
                            });
                        });

                        ui.separator();

                        // IMF Transcript
                        ui.allocate_ui([half_width, available_height].into(), |ui| {
                            ui.strong("🌐 IMF Session Log");
                            egui::ScrollArea::vertical().id_salt("imf_log_scroll").stick_to_bottom(true).show(ui, |ui| {
                                for log in &self.imf_logs {
                                    let bg = match log.entry_type.as_str() {
                                        "USER_INPUT" => egui::Color32::from_rgb(35, 30, 55),
                                        "PLANNER_RESPONSE" => egui::Color32::from_rgb(45, 40, 25),
                                        _ => egui::Color32::from_rgb(35, 35, 35),
                                    };
                                    egui::Frame::none().fill(bg).inner_margin(6.0).rounding(4.0).show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.strong(&log.entry_type);
                                            ui.label(format!("Step #{} • {}", log.step_index, log.created_at));
                                        });
                                        if let Some(content) = &log.content {
                                            ui.monospace(content);
                                        }
                                    });
                                    ui.add_space(3.0);
                                }
                            });
                        });
                    });
                }
                3 => {
                    // Active Background Tasks
                    ui.heading("⚙ Active Background Tasks (WEFT & IMF Fleet)");
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.strong("🧵 WEFT Running Tasks:");
                        egui::Grid::new("weft_tasks_grid").striped(true).min_col_width(120.0).show(ui, |ui| {
                            ui.strong("Task ID");
                            ui.strong("Log Size");
                            ui.strong("Recent Output Stream");
                            ui.end_row();
                            for t in &self.weft_tasks {
                                ui.label(&t.task_id);
                                ui.label(format!("{:.1} KB", t.size_bytes as f64 / 1024.0));
                                ui.monospace(&t.recent_line);
                                ui.end_row();
                            }
                        });

                        ui.add_space(16.0);
                        ui.strong("🌐 IMF Running Tasks:");
                        egui::Grid::new("imf_tasks_grid").striped(true).min_col_width(120.0).show(ui, |ui| {
                            ui.strong("Task ID");
                            ui.strong("Log Size");
                            ui.strong("Recent Output Stream");
                            ui.end_row();
                            for t in &self.imf_tasks {
                                ui.label(&t.task_id);
                                ui.label(format!("{:.1} KB", t.size_bytes as f64 / 1024.0));
                                ui.monospace(&t.recent_line);
                                ui.end_row();
                            }
                        });
                    });
                }
                4 => {
                    // Output Files & Artifacts
                    ui.horizontal(|ui| {
                        ui.heading("📁 Output Files & Exports");
                        ui.separator();
                        if ui.button("📂 Open in File Explorer").clicked() {
                            let _ = std::process::Command::new("xdg-open").arg(&self.output_dir).spawn();
                        }
                    });
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
            .with_inner_size([1280.0, 840.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("WEFT & IMF Studio - Unified AI Council & Live Cockpit"),
        ..Default::default()
    };
    eframe::run_native(
        "WEFT Studio",
        native_options,
        Box::new(|_cc| Ok(Box::new(WeftStudioApp::default()))),
    )
}
