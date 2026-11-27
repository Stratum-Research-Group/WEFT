import os

with open('tools/weft_studio/src/main.rs', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add fields to WeftStudioApp struct
fields = '''    // Stratum 95-Paper Corpus Explorer
    selected_paper_idx: usize,
    paper_search_query: String,
    cached_paper_content: String,
'''
if 'selected_paper_idx' not in text:
    text = text.replace('selected_tab: usize,', 'selected_tab: usize,\n' + fields)

# 2. Add defaults
defaults = '''            selected_paper_idx: 0,
            paper_search_query: String::new(),
            cached_paper_content: String::new(),
'''
if 'selected_paper_idx: 0,' not in text:
    text = text.replace('selected_tab: 1, // Default to Master Council tab', 'selected_tab: 1, // Default to Master Council tab\n' + defaults)

# 3. Add Tab selectable
tab_btn = 'ui.selectable_value(&mut self.selected_tab, 5, "📚 Stratum 95-Paper Monograph Explorer");'
if 'Stratum 95-Paper Monograph Explorer' not in text:
    text = text.replace(
        'ui.selectable_value(&mut self.selected_tab, 4, "📁 Sandbox & Artifacts Explorer");',
        'ui.selectable_value(&mut self.selected_tab, 4, "📁 Sandbox & Artifacts Explorer");\n                ' + tab_btn
    )

# 4. Add Tab 5 implementation
tab_impl = '''                5 => {
                    // Stratum 95-Paper Monograph & Figure Explorer
                    ui.heading("📚 Stratum Research Group & Synaptek Labs - 95-Paper Monograph Explorer");
                    ui.label("Certified 5-Pillar Corpus: Biophysical Parameters, Microarchitectures, Lean 4 Invariants, & Figures (#142 - #236)");
                    ui.separator();

                    let papers_dir = PathBuf::from("docs/PAPERS_REVISED");
                    let mut paper_files = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(&papers_dir) {
                        for e in entries.flatten() {
                            let name = e.file_name().to_string_lossy().to_string();
                            if name.starts_with("PAPER_") && name.ends_with(".md") {
                                paper_files.push(name);
                            }
                        }
                    }
                    paper_files.sort();

                    ui.horizontal(|ui| {
                        ui.label("Search Manuscripts:");
                        ui.text_edit_singleline(&mut self.paper_search_query);
                        ui.label(format!("Total: {} Papers", paper_files.len()));
                    });
                    ui.separator();

                    let filtered_papers: Vec<String> = paper_files
                        .into_iter()
                        .filter(|p| self.paper_search_query.is_empty() || p.to_lowercase().contains(&self.paper_search_query.to_lowercase()))
                        .collect();

                    ui.columns(2, |cols| {
                        // Left: Manuscript List
                        cols[0].strong("Manuscript Index");
                        egui::ScrollArea::vertical().id_salt("papers_list_scroll").max_height(600.0).show(&mut cols[0], |ui| {
                            for (idx, p_name) in filtered_papers.iter().enumerate() {
                                let is_selected = self.selected_paper_idx == idx;
                                if ui.selectable_label(is_selected, p_name).clicked() {
                                    self.selected_paper_idx = idx;
                                    let full_p = papers_dir.join(p_name);
                                    self.cached_paper_content = std::fs::read_to_string(&full_p).unwrap_or_default();
                                }
                            }
                        });

                        // Right: Content & Associated Proofs / Figures
                        cols[1].strong("Manuscript Viewer & Formal Invariants");
                        if self.cached_paper_content.is_empty() && !filtered_papers.is_empty() {
                            if let Some(first_p) = filtered_papers.get(self.selected_paper_idx) {
                                let full_p = papers_dir.join(first_p);
                                self.cached_paper_content = std::fs::read_to_string(&full_p).unwrap_or_default();
                            }
                        }

                        if let Some(curr_file) = filtered_papers.get(self.selected_paper_idx) {
                            cols[1].label(format!("Active File: {}", curr_file));

                            let num_str = curr_file.split('_').nth(1).unwrap_or("");
                            let fig1_png = format!("docs/figures/figure_paper_{}.png", num_str);
                            let fig2_png = format!("docs/figures/figure_paper_{}_validation.png", num_str);

                            cols[1].horizontal(|ui| {
                                if std::path::Path::new(&fig1_png).exists() {
                                    ui.label("Fig 1 (PNG/SVG): OK");
                                }
                                if std::path::Path::new(&fig2_png).exists() {
                                    ui.label("Fig 2 (Validation): OK");
                                }
                            });
                        }

                        egui::ScrollArea::vertical().id_salt("paper_content_scroll").max_height(550.0).show(&mut cols[1], |ui| {
                            ui.add_sized(
                                [ui.available_width(), 540.0],
                                egui::TextEdit::multiline(&mut self.cached_paper_content)
                                    .font(egui::TextStyle::Monospace)
                                    .interactive(false),
                            );
                        });
                    });
                }
'''
if '5 => {' not in text:
    text = text.replace('                _ => {}', tab_impl + '                _ => {}')

with open('tools/weft_studio/src/main.rs', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated main.rs successfully!')
