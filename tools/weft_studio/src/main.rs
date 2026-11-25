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
    #[allow(dead_code)]
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
struct DebateTurn {
    speaker: String,
    role_type: String, // "Proponent", "Skeptic/Adversary", "Synthesizer/Judge", "Architect"
    content: String,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MasterCouncilSession {
    instruction: String,
    domain: String,
    turns: Vec<DebateTurn>,
    final_verdict: Option<String>,
    sandbox_files_created: Vec<String>,
    is_debating: bool,
    status: String,
}

struct WeftStudioApp {
    // Model & domain selection
    selected_model_idx: usize,
    selected_domain: String,
    chat_input: String,
    imf_chat_history: Vec<ChatMessage>,
    weft_chat_history: Vec<ChatMessage>,

    // Master Council State
    master_council_sessions: Vec<MasterCouncilSession>,
    council_instruction_input: String,

    // Antigravity logs state
    weft_transcript_path: PathBuf,
    imf_transcript_path: PathBuf,
    weft_logs: Vec<LogEntry>,
    imf_logs: Vec<LogEntry>,
    weft_tasks: Vec<ActiveTask>,
    imf_tasks: Vec<ActiveTask>,

    output_dir: PathBuf,
    sandbox_dir: PathBuf,
    selected_tab: usize, // 0: 💬 Split Chat, 1: 🏛 Master Council (Protocol Debate & Sandbox), 2: 📜 Antigravity Dual Logs, 3: ⚙ Background Tasks, 4: 📁 Output & Sandbox Browser
    status_msg: String,
    auto_refresh: bool,
    is_generating: bool,

    // Async Channels
    tx: Sender<(String, ChatMessage, bool)>,
    rx: Receiver<(String, ChatMessage, bool)>,
    council_tx: Sender<(usize, Option<DebateTurn>, Option<String>, Vec<String>, bool, String)>,
    council_rx: Receiver<(usize, Option<DebateTurn>, Option<String>, Vec<String>, bool, String)>,
}

const MODELS: &[(&str, &str, BackendType)] = &[
    // Primary Active Local Engines
    ("Qwen 2.5 (14B High-Speed)", "qwen2.5:14b", BackendType::Ollama("qwen2.5:14b")),
    ("Qwen 3.8 (27B Frontier Reasoning)", "qwen3.8-27b:latest", BackendType::Ollama("qwen3.8-27b:latest")),
    ("Qwen 2.5-VL (7B Vision-Language)", "qwen2.5vl:7b", BackendType::Ollama("qwen2.5vl:7b")),
    ("WEFT Native GPU (Qwen2.5-0.5B Aligned)", "weft-qwen-instruct-0.5b", BackendType::WeftLivePort("http://127.0.0.1:8089")),

    // Full Resident Sub-Bit Fleet Registry (/run/media/akarta/AI/models/)
    ("WEFT DeepSeek Coder V2 Lite (16B MoE • 317.9 MB)", "deepseek-coder-v2-lite-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT DeepSeek R1 Distill Qwen (32B CoT • 147.9 MB)", "deepseek-r1-distill-qwen-32b-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Goedel-Prover Formal Math (32B • 148.6 MB)", "goedel-prover-32b-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Llama-3.3 Instruct (70B Dense • 220.6 MB)", "llama-3.3-70b-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Mixtral 8x7B Routing MoE (47B • 260.0 MB)", "mixtral-8x7b-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Qwen 2.5 Polyglot (72B Frontier • 230.1 MB)", "qwen-2.5-72b-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Qwen 2.5 Math Olympiad (72B • 229.5 MB)", "qwen2.5-math-72b-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Qwen 2.5 Coder Speculative (32B • 147.7 MB)", "qwen2.5-coder-32b-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Qwen 2.5 Coder Middle Draft (14B • 78.1 MB)", "qwen2.5-coder-14b-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Qwen 2.5 Coder Fast Draft (7B • 48.0 MB)", "qwen2.5-coder-7b-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Qwen2-VL Spatial VLM (7B • 59.4 MB)", "qwen2-vl-7b-subbit", BackendType::Ollama("qwen2.5vl:7b")),
    ("WEFT Whisper Large V3 ASR (1.55B • 60.4 MB)", "whisper-large-v3-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Gemma 4 Agent Tool Calling (26B MoE • 113.0 MB)", "gemma-4-26B-A4B-it-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Gemma 4 Edge Multi-Bitplane (E4B • 103.1 MB)", "gemma-4-E4B-it-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Gemma 4 Edge Compact (E2B • 67.2 MB)", "gemma-4-E2B-it-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Gemma 2 Dense (27B • 20.4 MB)", "gemma-2-27b-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Mistral Small Polyglot (24B • 16.6 MB)", "mistral-small-24b-subbit", BackendType::Ollama("qwen2.5:14b")),
    ("WEFT Llama-3.1 Hyperscale (405B Out-of-Core • 4.1 KB)", "llama-3.1-405b-instruct-subbit", BackendType::Ollama("qwen3.8-27b:latest")),
    ("WEFT Frontier Sparse MoE (1.04T / 128 Experts • 6.1 GB NVMe Shards)", "weft-frontier-1.04T-moe", BackendType::Ollama("qwen3.8-27b:latest")),
];

const EXPERT_DOMAINS: &[&str] = &[
    "System Architecture & Distributed Kernels",
    "Formal Lean 4 Invariant Verification",
    "Algebraic Geometry & Diophantine",
    "Blackwell Triton MoE Acceleration",
    "Speculative Decoding & Out-of-Core Streaming",
    "General Intelligence & Synthesis",
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
        let sandbox_folder = PathBuf::from("/home/akarta/Projects/0.WiP/Weft/tools/weft_studio/sandbox");
        std::fs::create_dir_all(&output_folder).ok();
        std::fs::create_dir_all(&sandbox_folder).ok();

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
            master_council_sessions: Vec::new(),
            council_instruction_input: String::new(),

            weft_transcript_path: brain_base.join(weft_conv_id).join(".system_generated/logs/transcript.jsonl"),
            imf_transcript_path: brain_base.join(imf_conv_id).join(".system_generated/logs/transcript.jsonl"),
            weft_logs: Vec::new(),
            imf_logs: Vec::new(),
            weft_tasks: Vec::new(),
            imf_tasks: Vec::new(),

            output_dir: output_folder,
            sandbox_dir: sandbox_folder,
            selected_tab: 1, // Default to Master Council tab
            status_msg: "Master Council Protocol Initialized (Sandbox Active)".to_string(),
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
        self.weft_logs = Self::read_transcript_tail(&self.weft_transcript_path);
        self.imf_logs = Self::read_transcript_tail(&self.imf_transcript_path);

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
            .timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| e.to_string())?;

        match backend {
            BackendType::Ollama(model_tag) => {
                let payload = serde_json::json!({
                    "model": model_tag,
                    "prompt": format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", system_prompt, user_prompt),
                    "stream": false,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 768
                    }
                });

                let res = client.post("http://127.0.0.1:11434/api/generate")
                    .json(&payload)
                    .send()
                    .map_err(|e| format!("Ollama connection error: {}", e))?;

                let val = res.json::<serde_json::Value>()
                    .map_err(|e| format!("JSON parse error: {}", e))?;

                if let Some(r) = val["response"].as_str() {
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

    fn dispatch_master_council_debate(&mut self) {
        if self.council_instruction_input.trim().is_empty() || self.is_generating {
            return;
        }

        let instruction = self.council_instruction_input.trim().to_string();
        self.council_instruction_input.clear();
        let domain = self.selected_domain.clone();
        let session_idx = self.master_council_sessions.len();

        let session = MasterCouncilSession {
            instruction: instruction.clone(),
            domain: domain.clone(),
            turns: Vec::new(),
            final_verdict: None,
            sandbox_files_created: Vec::new(),
            is_debating: true,
            status: "Initiating Protocol Phase 1: Proponent Thesis...".to_string(),
        };

        self.master_council_sessions.push(session);
        self.is_generating = true;
        self.status_msg = "Master Council Debating & Designing in Sandbox...".to_string();

        let council_tx = self.council_tx.clone();
        let sandbox_dir = self.sandbox_dir.clone();

        // Spawn Multi-Turn Protocol Thread
        thread::spawn(move || {
            let mut created_files = Vec::new();

            // PHASE 1: Proponent Thesis (Qwen 2.5 14B High-Speed)
            let proponent_model = BackendType::Ollama("qwen2.5:14b");
            let p1_sys = format!(
                "You are the ADVOCATE & LEAD DESIGNER in the Fleet Council for {}. Propose a concrete, technical design and solution to accomplish the user's goal. Lay out the architectural components clearly with pros.",
                domain
            );
            let p1_text = match Self::query_model(proponent_model, &p1_sys, &instruction) {
                Ok(t) => t,
                Err(e) => format!("Proponent failed to generate: {}", e),
            };

            let t1 = DebateTurn {
                speaker: "Advocate (Qwen 2.5 14B)".to_string(),
                role_type: "Proponent Thesis".to_string(),
                content: p1_text.clone(),
                timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
            };
            let _ = council_tx.send((session_idx, Some(t1), None, created_files.clone(), true, "Phase 2: Skeptic / Adversary Analysis...".to_string()));

            // PHASE 2: Skeptic / Adversary Analysis (Qwen 3.8 27B Deep Reasoning)
            let skeptic_model = BackendType::Ollama("qwen3.8-27b:latest");
            let p2_sys = format!(
                "You are the CRITIC & SAFETY ADVERSARY in the Fleet Council for {}. Rigorously analyze the proposed design below. Highlight edge cases, performance bottlenecks, race conditions, memory leaks, and architectural risks. Weigh pros vs cons objectively.",
                domain
            );
            let p2_prompt = format!("Original User Instruction:\n{}\n\nProposed Design:\n{}", instruction, p1_text);
            let p2_text = match Self::query_model(skeptic_model, &p2_sys, &p2_prompt) {
                Ok(t) => t,
                Err(e) => format!("Skeptic failed to generate: {}", e),
            };

            let t2 = DebateTurn {
                speaker: "Adversary (Qwen 3.8 27B CoT)".to_string(),
                role_type: "Skeptic Critique & Risk Analysis".to_string(),
                content: p2_text.clone(),
                timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
            };
            let _ = council_tx.send((session_idx, Some(t2), None, created_files.clone(), true, "Phase 3: Sandbox Implementation & Artifact Authoring...".to_string()));

            // PHASE 3: Sandbox Builder & Implementation (WEFT Native GPU / Coder)
            let builder_model = BackendType::WeftLivePort("http://127.0.0.1:8089");
            let p3_sys = "You are the SYSTEMS BUILDER for the Fleet Council. Write a concrete, production-ready implementation, script, or specification incorporating both the proposed design and the critique.";
            let p3_prompt = format!("Instruction:\n{}\n\nDesign:\n{}\n\nCritique:\n{}\n\nProvide the code or specification.", instruction, p1_text, p2_text);
            let p3_text = match Self::query_model(builder_model, p3_sys, &p3_prompt) {
                Ok(t) => t,
                Err(_) => {
                    // Fallback to Ollama 14B if live server busy
                    Self::query_model(BackendType::Ollama("qwen2.5:14b"), p3_sys, &p3_prompt).unwrap_or_else(|e| format!("Builder error: {}", e))
                }
            };

            // Write implementation into Sandbox file
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            let sandbox_filename = format!("council_artifact_{}.rs", ts);
            let sandbox_path = sandbox_dir.join(&sandbox_filename);
            let sandbox_content = format!(
                "// ================================================================\n// MASTER COUNCIL SANDBOX ARTIFACT\n// Target Domain: {}\n// Instruction: {}\n// Generated: {}\n// ================================================================\n\n{}\n",
                domain, instruction, ts, p3_text
            );
            if std::fs::write(&sandbox_path, sandbox_content).is_ok() {
                created_files.push(sandbox_filename.clone());
            }

            let t3 = DebateTurn {
                speaker: "Systems Builder (WEFT Local GPU)".to_string(),
                role_type: "Sandbox Implementation".to_string(),
                content: format!("Wrote implementation artifact to sandbox: {}\n\n{}", sandbox_filename, p3_text),
                timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
            };
            let _ = council_tx.send((session_idx, Some(t3), None, created_files.clone(), true, "Phase 4: Synthesis & Final Binding Verdict...".to_string()));

            // PHASE 4: Synthesis & Final Verdict (Chief Justice / Qwen 3.8 27B)
            let judge_model = BackendType::Ollama("qwen3.8-27b:latest");
            let p4_sys = "You are the CHIEF ARBITER of the Fleet Council. Review the proposal, the critique, and the sandbox implementation. Render the FINAL VERDICT with clear consensus, verified invariants, and concrete next actions.";
            let p4_prompt = format!(
                "User Instruction:\n{}\n\nAdvocate Proposal:\n{}\n\nAdversary Critique:\n{}\n\nBuilder Artifact:\n{}\n\nIssue the Final Verdict.",
                instruction, p1_text, p2_text, p3_text
            );
            let verdict_text = match Self::query_model(judge_model, p4_sys, &p4_prompt) {
                Ok(t) => t,
                Err(e) => format!("Chief Arbiter failed to issue verdict: {}", e),
            };

            // Completed Protocol Session
            let _ = council_tx.send((session_idx, None, Some(verdict_text), created_files, false, "Protocol Completed - Final Verdict Issued".to_string()));
        });
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

            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            let out_path = self.output_dir.join(format!("{}_chat_{}.json", target, ts));
            let data = if target == "imf" { &self.imf_chat_history } else { &self.weft_chat_history };
            if let Ok(file_data) = serde_json::to_string_pretty(data) {
                let _ = std::fs::write(&out_path, file_data);
            }
        }

        while let Ok((s_idx, turn_opt, verdict_opt, sandbox_files, is_active, status)) = self.council_rx.try_recv() {
            if s_idx < self.master_council_sessions.len() {
                let session = &mut self.master_council_sessions[s_idx];
                if let Some(turn) = turn_opt {
                    session.turns.push(turn);
                }
                if let Some(verdict) = verdict_opt {
                    session.final_verdict = Some(verdict);
                }
                session.sandbox_files_created = sandbox_files;
                session.is_debating = is_active;
                session.status = status.clone();

                if !is_active {
                    self.is_generating = false;
                    self.status_msg = "Master Council Verdict Ready".to_string();

                    // Persist session to disk
                    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
                    let out_path = self.output_dir.join(format!("master_council_debate_{}.json", ts));
                    if let Ok(file_data) = serde_json::to_string_pretty(&session) {
                        let _ = std::fs::write(&out_path, file_data);
                    }
                } else {
                    self.status_msg = status;
                }
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
                ui.heading("⚡ WEFT & IMF Studio | Autonomous Fleet Council & Sandbox");
                ui.separator();
                if ui.button("🔄 Sync Transcripts & Tasks").clicked() {
                    self.reload_logs();
                }
                ui.checkbox(&mut self.auto_refresh, "Auto-Poll (1s)");
                ui.separator();
                ui.label(format!("Council Engine: {}", self.status_msg));
            });

            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.selected_tab, 1, "🏛 Master Council (Protocol Debate & Sandbox)");
                ui.selectable_value(&mut self.selected_tab, 0, "💬 Split Chat (WEFT | IMF)");
                ui.selectable_value(&mut self.selected_tab, 2, "📜 Antigravity Dual Logs");
                ui.selectable_value(&mut self.selected_tab, 3, "⚙ Active Background Tasks");
                ui.selectable_value(&mut self.selected_tab, 4, "📁 Sandbox & Artifacts Explorer");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.selected_tab {
                1 => {
                    // MASTER COUNCIL TAB
                    ui.horizontal(|ui| {
                        ui.heading("🏛 Sub-Bit Fleet Master Council");
                        ui.label("| Multi-model debate protocol: Advocate ➔ Adversary ➔ Builder ➔ Arbiter");
                        ui.separator();
                        ui.label("Specialist Domain:");
                        egui::ComboBox::from_id_salt("council_domain_select")
                            .selected_text(&self.selected_domain)
                            .show_ui(ui, |ui| {
                                for d in EXPERT_DOMAINS {
                                    ui.selectable_value(&mut self.selected_domain, d.to_string(), *d);
                                }
                            });
                        if ui.button("📂 Open Sandbox Folder").clicked() {
                            let _ = std::process::Command::new("xdg-open").arg(&self.sandbox_dir).spawn();
                        }
                    });
                    ui.separator();

                    let available_height = ui.available_height() - 60.0;
                    egui::ScrollArea::vertical()
                        .id_salt("master_council_scroll")
                        .max_height(available_height)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            if self.master_council_sessions.is_empty() {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(25, 30, 40))
                                    .inner_margin(16.0)
                                    .rounding(8.0)
                                    .show(ui, |ui| {
                                        ui.heading("Council Comms Protocol Ready");
                                        ui.label("Give an instruction below. The Master Council will autonomously execute the 4-phase debate protocol:");
                                        ui.label("1. 📢 Advocate Proposal (Qwen 14B High-Speed): Constructs the core architecture and pros.");
                                        ui.label("2. 🔍 Adversary Critique (Qwen 3.8 27B CoT): Challenges bottlenecks, risks, cons, and edge cases.");
                                        ui.label("3. 🛠 Systems Builder (WEFT Local GPU): Writes and saves implementation code into the Sandbox.");
                                        ui.label("4. ⚖ Chief Arbiter (Qwen 3.8 27B): Weighs all logic and issues the binding Final Verdict.");
                                    });
                            }

                            for session in &self.master_council_sessions {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(20, 25, 35))
                                    .inner_margin(12.0)
                                    .rounding(6.0)
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.strong(format!("📋 Instruction: \"{}\"", session.instruction));
                                            ui.separator();
                                            ui.label(format!("Domain: [{}] • Status: {}", session.domain, session.status));
                                        });

                                        if !session.sandbox_files_created.is_empty() {
                                            ui.add_space(4.0);
                                            ui.horizontal(|ui| {
                                                ui.strong("🛠 Sandbox Artifacts Created:");
                                                for f in &session.sandbox_files_created {
                                                    ui.monospace(format!("📁 {}", f));
                                                }
                                            });
                                        }

                                        ui.separator();

                                        // Debate Turns
                                        for turn in &session.turns {
                                            let (badge_color, role_icon) = match turn.role_type.as_str() {
                                                "Proponent Thesis" => (egui::Color32::from_rgb(30, 60, 40), "📢"),
                                                "Skeptic Critique & Risk Analysis" => (egui::Color32::from_rgb(60, 35, 30), "🔍"),
                                                "Sandbox Implementation" => (egui::Color32::from_rgb(30, 45, 65), "🛠"),
                                                _ => (egui::Color32::from_rgb(45, 45, 55), "💬"),
                                            };

                                            egui::Frame::none()
                                                .fill(badge_color)
                                                .inner_margin(10.0)
                                                .rounding(6.0)
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.strong(format!("{} {} - {}", role_icon, turn.speaker, turn.role_type));
                                                        ui.label(format!("• {}", turn.timestamp));
                                                    });
                                                    ui.add_space(4.0);
                                                    ui.label(&turn.content);
                                                });
                                            ui.add_space(6.0);
                                        }

                                        // Final Verdict
                                        if let Some(verdict) = &session.final_verdict {
                                            egui::Frame::none()
                                                .fill(egui::Color32::from_rgb(40, 50, 25))
                                                .inner_margin(12.0)
                                                .rounding(6.0)
                                                .stroke(egui::Stroke::new(1.5, egui::Color32::from_rgb(200, 180, 50)))
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.heading("⚖ FINAL COUNCIL VERDICT & CONSENSUS");
                                                    });
                                                    ui.separator();
                                                    ui.label(verdict);
                                                });
                                        }
                                    });
                                ui.add_space(12.0);
                            }
                        });

                    ui.separator();
                    // Master Council Input Bar
                    ui.horizontal(|ui| {
                        let text_edit = ui.add_sized(
                            [ui.available_width() - 220.0, 36.0],
                            egui::TextEdit::singleline(&mut self.council_instruction_input).hint_text("Give an instruction for the Council to debate, design, and implement..."),
                        );

                        let send_clicked = ui.button("⚡ Convene Master Council").clicked();
                        if (send_clicked || (text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))) && !self.is_generating {
                            self.dispatch_master_council_debate();
                        }
                    });
                }
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
                    });
                    ui.separator();

                    let available_height = ui.available_height() - 50.0;
                    let half_width = (ui.available_width() - 20.0) / 2.0;

                    ui.horizontal(|ui| {
                        // WEFT Column
                        ui.allocate_ui([half_width, available_height].into(), |ui| {
                            ui.heading("🧵 WEFT Chat Column");
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
                            egui::TextEdit::singleline(&mut self.chat_input).hint_text("Type a message to send directly to WEFT or IMF..."),
                        );

                        if ui.button("🧵 Send to WEFT").clicked() || (text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                            self.dispatch_chat("weft");
                        }
                        if ui.button("🌐 Send to IMF").clicked() {
                            self.dispatch_chat("imf");
                        }
                    });
                }
                2 => {
                    // Dual Antigravity Logs
                    ui.heading("📜 Antigravity Live Transcripts (WEFT vs IMF Side-by-Side)");
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
                                            if let Some(content) = &log.content {
                                                if ui.small_button("⏩ Fork into WEFT Chat").clicked() {
                                                    self.chat_input = format!("Continuing from Antigravity Step #{}: {}", log.step_index, content);
                                                    self.selected_tab = 0;
                                                }
                                            }
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
                                            if let Some(content) = &log.content {
                                                if ui.small_button("⏩ Fork into IMF Chat").clicked() {
                                                    self.chat_input = format!("Continuing from IMF Step #{}: {}", log.step_index, content);
                                                    self.selected_tab = 0;
                                                }
                                            }
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
                    // Output & Sandbox Artifacts Browser
                    ui.horizontal(|ui| {
                        ui.heading("📁 Sandbox & Output Files");
                        ui.separator();
                        if ui.button("📂 Open Sandbox Folder").clicked() {
                            let _ = std::process::Command::new("xdg-open").arg(&self.sandbox_dir).spawn();
                        }
                        if ui.button("📂 Open Output Folder").clicked() {
                            let _ = std::process::Command::new("xdg-open").arg(&self.output_dir).spawn();
                        }
                    });
                    ui.separator();

                    ui.strong("🛠 Council Sandbox Artifacts:");
                    egui::ScrollArea::vertical().id_salt("sandbox_scroll").max_height(200.0).show(ui, |ui| {
                        if let Ok(entries) = std::fs::read_dir(&self.sandbox_dir) {
                            for e in entries.flatten() {
                                ui.horizontal(|ui| {
                                    ui.label("⚙");
                                    ui.strong(e.file_name().to_string_lossy().to_string());
                                    if let Ok(meta) = e.metadata() {
                                        ui.label(format!("{:.1} KB", meta.len() as f64 / 1024.0));
                                    }
                                });
                            }
                        }
                    });

                    ui.separator();
                    ui.strong("📄 Conversation & Council Snapshots:");
                    egui::ScrollArea::vertical().id_salt("output_scroll").show(ui, |ui| {
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
            .with_inner_size([1320.0, 860.0])
            .with_min_inner_size([880.0, 620.0])
            .with_title("WEFT & IMF Studio - Sub-Bit Fleet Master Council & Sandbox Cockpit"),
        ..Default::default()
    };
    eframe::run_native(
        "WEFT Studio",
        native_options,
        Box::new(|_cc| Ok(Box::new(WeftStudioApp::default()))),
    )
}
