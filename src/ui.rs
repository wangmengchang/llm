use crate::app::{App, Modal, Screen, SettingsRow, Tone, SETTINGS_ROWS};
use crate::backend::binary_available;
use crate::config::primary_models_dir;
use crate::fields::TABS;
use crate::models::{human_size, ModelKind, ModelSource};
use crate::server::{format_uptime, ServerStatus};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Clear, List, ListItem, Paragraph, Tabs, Wrap};
use ratatui::Frame;

const ACCENT: Color = Color::Cyan;
const GOOD: Color = Color::Green;
const WARN: Color = Color::Yellow;
const BAD: Color = Color::Red;
const MUTED: Color = Color::DarkGray;
const TEXT: Color = Color::Gray;
const SELECT_BG: Color = Color::Rgb(28, 42, 58);
const PANEL_BG: Color = Color::Rgb(20, 24, 30);

pub fn draw(frame: &mut Frame, app: &mut App) {
    app.ui = Default::default();
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(PANEL_BG)), area);

    let chunks = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(0),
        Constraint::Length(2),
    ])
    .split(area);
    app.ui.body_height = chunks[1].height;

    draw_header(frame, app, chunks[0]);
    match app.screen {
        Screen::Models => draw_models(frame, app, chunks[1]),
        Screen::Configure => draw_configure(frame, app, chunks[1]),
        Screen::Running => draw_running(frame, app, chunks[1]),
        Screen::Settings => draw_settings(frame, app, chunks[1]),
    }
    draw_footer(frame, app, chunks[2]);

    if app.show_help {
        draw_help(frame, app);
    }
    if app.modal.is_some() {
        draw_modal(frame, app);
    }
    if app.editing.is_some() {
        draw_editor(frame, app);
    }
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let block = panel_block("", MUTED);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 2 {
        return;
    }
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(inner);
    let wide = inner.width >= 100;
    let right_width = if wide { 48 } else { 42 };

    let left =
        Layout::horizontal([Constraint::Min(20), Constraint::Length(right_width)]).split(rows[0]);
    let mut title = vec![
        Span::styled(
            " llm ",
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Local LLM Manager", Style::default().fg(Color::White)),
    ];
    if wide {
        title.push(Span::styled(
            "   friendly local model runner",
            Style::default().fg(MUTED),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(title)), left[0]);

    let (server_text, server_style) = server_badge(app);
    let right = Line::from(vec![
        Span::styled(
            app.cfg.backend.label().to_string(),
            Style::default().fg(ACCENT),
        ),
        Span::styled(
            if wide {
                format!("  |  {}  |  ", crate::models::count_label(app.models.len()))
            } else {
                format!(" | {} | ", crate::models::count_label(app.models.len()))
            },
            Style::default().fg(MUTED),
        ),
        Span::styled(server_text, server_style),
    ]);
    frame.render_widget(Paragraph::new(right).alignment(Alignment::Right), left[1]);

    let second =
        Layout::horizontal([Constraint::Min(20), Constraint::Length(right_width)]).split(rows[1]);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            if wide || inner.width >= 78 {
                " Everything stays on this computer."
            } else {
                " Stays on this computer."
            },
            Style::default().fg(MUTED),
        ))),
        second[0],
    );

    let program = app.cfg.binary(app.cfg.backend);
    let found = binary_available(program);
    let binary_line = Line::from(vec![
        Span::styled("backend: ", Style::default().fg(MUTED)),
        Span::styled(
            if found { "found" } else { "not found" },
            Style::default().fg(if found { GOOD } else { WARN }),
        ),
        Span::styled(format!("  {program}"), Style::default().fg(MUTED)),
    ]);
    frame.render_widget(
        Paragraph::new(binary_line).alignment(Alignment::Right),
        second[1],
    );
}

fn server_badge(app: &App) -> (String, Style) {
    match app.server.status() {
        ServerStatus::Idle => ("no server".to_string(), Style::default().fg(MUTED)),
        ServerStatus::Starting => {
            let port = app.server.info().map(|info| info.port).unwrap_or(0);
            (
                format!("loading :{port}"),
                Style::default().fg(WARN).add_modifier(Modifier::BOLD),
            )
        }
        ServerStatus::Ready => {
            let port = app.server.info().map(|info| info.port).unwrap_or(0);
            (
                format!("server ready :{port}"),
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            )
        }
        ServerStatus::Exited(_) => ("server stopped".to_string(), Style::default().fg(BAD)),
    }
}

fn draw_models(frame: &mut Frame, app: &mut App, area: Rect) {
    if app.models.is_empty() {
        draw_onboarding(frame, app, area);
        return;
    }

    let cols = Layout::horizontal([
        Constraint::Percentage(57),
        Constraint::Length(1),
        Constraint::Percentage(42),
    ])
    .split(area);
    draw_model_list(frame, app, cols[0]);
    draw_model_details(frame, app, cols[2]);
}

fn draw_model_list(frame: &mut Frame, app: &mut App, area: Rect) {
    let block = panel_block(&format!(" Models ({}) ", app.models.len()), ACCENT);
    let inner = block.inner(area);

    let items: Vec<ListItem> = app
        .models
        .iter()
        .map(|model| {
            let mut first = vec![Span::styled(
                model.name.clone(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )];
            if model.source != ModelSource::Local {
                first.push(Span::raw("  "));
                first.push(Span::styled(
                    format!("[{}]", model.source.label()),
                    Style::default().fg(Color::Magenta),
                ));
            }
            let tags = model.tags();
            let mut second = vec![Span::styled("  ", Style::default())];
            if tags.is_empty() {
                let label = match model.kind {
                    ModelKind::Gguf => "GGUF model",
                    ModelKind::HfDir => "model folder",
                };
                second.push(Span::styled(label.to_string(), Style::default().fg(MUTED)));
            } else {
                second.push(Span::styled(tags.join("  ·  "), Style::default().fg(MUTED)));
            }
            second.push(Span::styled(
                format!("   {}", human_size(model.size_bytes)),
                Style::default().fg(MUTED),
            ));
            ListItem::new(vec![Line::from(first), Line::from(second)])
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(SELECT_BG).fg(Color::White))
        .highlight_symbol("▌ ");

    let mut state = app.models_state.clone();
    frame.render_stateful_widget(list, area, &mut state);
    app.models_state = state;
    app.ui.models_list = Some(inner);
}

fn draw_model_details(frame: &mut Frame, app: &App, area: Rect) {
    let block = panel_block(" Selected model ", ACCENT);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(model) = app.selected_model() else {
        return;
    };
    let backend_ok = binary_available(app.cfg.binary(app.cfg.backend));
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        model.name.clone(),
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(vec![
        Span::styled(
            model.source.label().to_string(),
            Style::default().fg(Color::Magenta),
        ),
        Span::styled(
            format!("  ·  {}", human_size(model.size_bytes)),
            Style::default().fg(MUTED),
        ),
    ]));
    if let Some(meta) = &model.meta_name {
        lines.push(Line::from(vec![
            Span::styled("model name  ", Style::default().fg(MUTED)),
            Span::raw(meta.clone()),
        ]));
    }
    let mut info = Vec::new();
    if let Some(params) = &model.params {
        info.push(params.clone());
    }
    if let Some(quant) = &model.quant {
        info.push(quant.clone());
    }
    if let Some(ctx) = model.context_label() {
        info.push(format!("{ctx} context"));
    }
    if let Some(arch) = &model.arch {
        info.push(arch.clone());
    }
    if !info.is_empty() {
        lines.push(Line::from(Span::styled(
            info.join("  ·  "),
            Style::default().fg(TEXT),
        )));
    }
    let compact = inner.height < 22;
    if !compact {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "FILE",
            Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            model.path.to_string_lossy().to_string(),
            Style::default().fg(TEXT),
        )));
    }
    lines.push(Line::from(""));

    let settings = app.cfg.settings_for(&model.path);
    lines.push(Line::from(Span::styled(
        "READY TO RUN WITH",
        Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
    )));
    lines.push(kv("Preset", settings.preset.label(), ACCENT));
    lines.push(kv(
        "Context",
        &if settings.context_size == 0 {
            "Auto (default)".to_string()
        } else {
            format!(
                "{} tokens",
                crate::config::format_token_count(settings.context_size as u64)
            )
        },
        TEXT,
    ));
    lines.push(kv(
        "GPU",
        &match settings.gpu_layers {
            -1 => "Auto".to_string(),
            -2 => "All layers".to_string(),
            n => format!("{n} layers"),
        },
        TEXT,
    ));
    lines.push(kv(
        "Backend",
        &format!(
            "{} ({})",
            app.cfg.backend.label(),
            if backend_ok { "found" } else { "not found" }
        ),
        if backend_ok { GOOD } else { WARN },
    ));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            "Enter",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" tune settings", Style::default().fg(TEXT)),
        Span::styled("    ", Style::default()),
        Span::styled(
            "L",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" run now", Style::default().fg(TEXT)),
    ]));
    lines.push(Line::from(""));
    if !backend_ok {
        lines.push(Line::from(Span::styled(
            format!("{} is not installed yet.", app.cfg.binary(app.cfg.backend)),
            Style::default().fg(WARN).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            app.cfg.backend.install_hint().to_string(),
            Style::default().fg(TEXT),
        )));
    } else {
        let blurb = match app.cfg.backend {
            crate::config::Backend::Vllm => {
                "llm will start a private server with an OpenAI-compatible API."
            }
            crate::config::Backend::LlamaCpp => {
                "llm will start a private server and open a chat page."
            }
        };
        lines.push(Line::from(Span::styled(blurb, Style::default().fg(TEXT))));
        lines.push(Line::from(Span::styled(
            "Your prompts never leave this computer.",
            Style::default().fg(MUTED),
        )));
    }

    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_onboarding(frame: &mut Frame, app: &App, area: Rect) {
    let horizontal = Layout::horizontal([
        Constraint::Percentage(5),
        Constraint::Percentage(90),
        Constraint::Percentage(5),
    ])
    .split(area);

    let block = panel_block(" Welcome to llm ", ACCENT);
    let inner = block.inner(horizontal[1]);
    frame.render_widget(block, horizontal[1]);

    let backend = app.cfg.backend;
    let program = app.cfg.binary(backend);
    let backend_line = match backend {
        crate::config::Backend::LlamaCpp => {
            "1. Add a .gguf model, or use models from Ollama or LM Studio."
        }
        crate::config::Backend::Vllm => {
            "1. Add a Hugging Face model folder (it contains config.json)."
        }
    };
    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            "Runs AI models on your computer. Nothing is uploaded.",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(backend_line, Style::default().fg(TEXT))),
        Line::from(Span::styled(
            "2. Pick a model. Enter tunes it, or L runs it.",
            Style::default().fg(TEXT),
        )),
        Line::from(Span::styled(
            match backend {
                crate::config::Backend::LlamaCpp => {
                    "3. Press o to open the chat page when the server is ready."
                }
                crate::config::Backend::Vllm => {
                    "3. Press o to open the API docs when the server is ready."
                }
            },
            Style::default().fg(TEXT),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Searched",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!(
                "  {} standard folders. Press s to add your own.",
                app.searched.len()
            ),
            Style::default().fg(MUTED),
        )),
    ];

    if !binary_available(program) {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("{program} is not installed yet."),
            Style::default().fg(WARN).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            backend.short_install_hint(),
            Style::default().fg(TEXT),
        )));
    }

    if app.models.is_empty() {
        lines.push(Line::from(""));
        match backend {
            crate::config::Backend::LlamaCpp => {
                lines.push(Line::from(Span::styled(
                    "No models found yet. Press d to open a GGUF download page.",
                    Style::default().fg(TEXT),
                )));
            }
            crate::config::Backend::Vllm => {
                lines.push(Line::from(Span::styled(
                    "No models found. vLLM does not use .gguf files.",
                    Style::default().fg(TEXT),
                )));
                lines.push(Line::from(Span::styled(
                    "Press s to switch back to llama.cpp for GGUF models.",
                    Style::default().fg(TEXT),
                )));
            }
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            "s",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" settings   ", Style::default().fg(TEXT)),
        Span::styled(
            "r",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" rescan   ", Style::default().fg(TEXT)),
        Span::styled(
            "o",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" open folder   ", Style::default().fg(TEXT)),
        Span::styled(
            "d",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" download   ", Style::default().fg(TEXT)),
        Span::styled(
            "?",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" help", Style::default().fg(TEXT)),
    ]));

    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_configure(frame: &mut Frame, app: &mut App, area: Rect) {
    let show_command = app.show_command && area.height >= 18;
    let chunks = if show_command {
        Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(7),
            Constraint::Length(8),
        ])
        .split(area)
    } else {
        Layout::vertical([Constraint::Length(3), Constraint::Min(8)]).split(area)
    };

    draw_tabs(frame, app, chunks[0]);

    let cols = Layout::horizontal([
        Constraint::Percentage(55),
        Constraint::Length(1),
        Constraint::Percentage(44),
    ])
    .split(chunks[1]);
    draw_field_list(frame, app, cols[0]);
    draw_field_help(frame, app, cols[2]);

    if show_command {
        draw_command_preview(frame, app, chunks[2]);
    }
}

fn draw_tabs(frame: &mut Frame, app: &mut App, area: Rect) {
    let model_name = app
        .selected_model()
        .map(|model| model.name.clone())
        .unwrap_or_else(|| "model".to_string());
    let block = panel_block(&format!(" Configure  {model_name} "), ACCENT);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cols =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).split(inner);
    let titles: Vec<Line> = TABS
        .iter()
        .map(|tab| Line::from(format!(" {tab} ")))
        .collect();
    let tabs = Tabs::new(titles)
        .select(app.cfg_tab)
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        )
        .divider(Span::styled(" ", Style::default()));
    frame.render_widget(tabs, cols[0]);
    app.ui.cfg_tabs = Some(cols[0]);

    let line = Line::from(vec![
        Span::styled(
            app.cfg.backend.label().to_string(),
            Style::default().fg(ACCENT),
        ),
        Span::styled("  1/2/3 or Tab", Style::default().fg(MUTED)),
    ]);
    frame.render_widget(Paragraph::new(line).alignment(Alignment::Right), cols[1]);
}

fn draw_field_list(frame: &mut Frame, app: &mut App, area: Rect) {
    let tab = TABS[app.cfg_tab];
    let block = panel_block(&format!(" {tab} "), MUTED);
    let inner = block.inner(area);

    let items: Vec<ListItem> = app
        .fields()
        .iter()
        .map(|field| {
            let value = app.field_value(*field);
            let value_style = match field.kind() {
                crate::fields::FieldKind::Bool => {
                    if value == "On" {
                        Style::default().fg(GOOD)
                    } else {
                        Style::default().fg(MUTED)
                    }
                }
                crate::fields::FieldKind::Text => Style::default().fg(TEXT),
                _ => Style::default().fg(ACCENT),
            };
            let label = format!("{:<22}", field.label());
            ListItem::new(Line::from(vec![
                Span::styled(label, Style::default().fg(Color::White)),
                Span::styled(value, value_style),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(SELECT_BG).fg(Color::White))
        .highlight_symbol("▌ ");
    let mut state = app.cfg_state.clone();
    frame.render_stateful_widget(list, area, &mut state);
    app.cfg_state = state;
    app.ui.cfg_list = Some(inner);
}

fn draw_field_help(frame: &mut Frame, app: &App, area: Rect) {
    let field = app.current_field();
    let title = field
        .map(|field| format!(" About: {} ", field.label()))
        .unwrap_or_else(|| " About ".to_string());
    let block = panel_block(&title, ACCENT);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(field) = field else {
        return;
    };
    let mut lines = vec![
        Line::from(Span::styled(field.help(), Style::default().fg(TEXT))),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "Recommended  ",
                Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                field.recommendation().to_string(),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(""),
    ];
    let how = match field.kind() {
        crate::fields::FieldKind::Bool => "Press Enter or Left/Right to toggle.",
        crate::fields::FieldKind::Cycle => "Press Left/Right to cycle. Enter goes forward.",
        crate::fields::FieldKind::Preset => {
            "Press Left/Right to apply a preset. It fills in safe values for you."
        }
        crate::fields::FieldKind::Text => "Press Enter to type a value.",
        _ => "Press Left/Right for small steps, or Enter to type an exact value.",
    };
    lines.push(Line::from(Span::styled(how, Style::default().fg(MUTED))));
    lines.push(Line::from(""));
    if field == crate::fields::Field::DraftModel {
        let candidates = app.draft_candidates();
        lines.push(Line::from(Span::styled(
            format!("Draft models available: {}", candidates.len()),
            Style::default().fg(TEXT),
        )));
        lines.push(Line::from(Span::styled(
            "None is fine. Speculation is optional.",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
    }
    if field == crate::fields::Field::Preset {
        lines.push(Line::from(Span::styled(
            app.config_settings.preset.description().to_string(),
            Style::default().fg(ACCENT),
        )));
        lines.push(Line::from(""));
    }
    lines.push(Line::from(Span::styled(
        "Tip: if something fails, lower Context length or set GPU layers to Auto.",
        Style::default().fg(MUTED),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            "L",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" launch with these settings    ", Style::default().fg(TEXT)),
        Span::styled(
            "Esc",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" back", Style::default().fg(TEXT)),
    ]));

    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_command_preview(frame: &mut Frame, app: &App, area: Rect) {
    let block = panel_block(" What will run (c to hide) ", WARN);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(model) = app.selected_model() else {
        return;
    };
    let program = app.cfg.binary(app.cfg.backend);
    let preview =
        crate::backend::build_plan(app.cfg.backend, &app.config_settings, &model.path, program);
    let lines = match preview {
        Ok(plan) => vec![
            Line::from(Span::styled(
                format!("$ {}", plan.command_line()),
                Style::default().fg(Color::White),
            )),
            Line::from(Span::styled(
                "Exact command. The port may change automatically if it is busy.",
                Style::default().fg(MUTED),
            )),
        ],
        Err(err) => vec![
            Line::from(Span::styled(
                format!("Not ready to launch: {err}"),
                Style::default().fg(WARN),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Fix the highlighted setting and the command will appear here.",
                Style::default().fg(MUTED),
            )),
        ],
    };
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_settings(frame: &mut Frame, app: &mut App, area: Rect) {
    let (list_pct, help_pct) = if area.width < 90 { (54, 45) } else { (48, 51) };
    let cols = Layout::horizontal([
        Constraint::Percentage(list_pct),
        Constraint::Length(1),
        Constraint::Percentage(help_pct),
    ])
    .split(area);

    let block = panel_block(" Settings ", ACCENT);
    let inner = block.inner(cols[0]);
    let items: Vec<ListItem> = SETTINGS_ROWS
        .iter()
        .map(|row| {
            let value = app.settings_value(*row);
            let value_style = match row {
                SettingsRow::Backend => Style::default().fg(ACCENT),
                SettingsRow::LlamaBin | SettingsRow::VllmBin => {
                    if value.ends_with("found") {
                        Style::default().fg(GOOD)
                    } else {
                        Style::default().fg(WARN)
                    }
                }
                SettingsRow::Rescan | SettingsRow::OpenConfig | SettingsRow::OpenModels => {
                    Style::default().fg(MUTED)
                }
                SettingsRow::ModelsDir => Style::default().fg(TEXT),
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<20}", row.label()),
                    Style::default().fg(Color::White),
                ),
                Span::styled(value, value_style),
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(SELECT_BG).fg(Color::White))
        .highlight_symbol("▌ ");
    let mut state = app.settings_state.clone();
    frame.render_stateful_widget(list, cols[0], &mut state);
    app.settings_state = state;
    app.ui.settings_list = Some(inner);

    let row = app.settings_row();
    let block = panel_block(&format!(" About: {} ", row.label()), ACCENT);
    let inner = block.inner(cols[2]);
    frame.render_widget(block, cols[2]);

    let mut lines = vec![
        Line::from(Span::styled(row.help(), Style::default().fg(TEXT))),
        Line::from(""),
    ];
    match row {
        SettingsRow::Backend => {
            lines.push(kv("Current", app.cfg.backend.label(), ACCENT));
            lines.push(kv(
                "llama-server",
                if binary_available(&app.cfg.llama_server) {
                    "found"
                } else {
                    "not found"
                },
                if binary_available(&app.cfg.llama_server) {
                    GOOD
                } else {
                    WARN
                },
            ));
            lines.push(kv(
                "vLLM",
                if binary_available(&app.cfg.vllm) {
                    "found"
                } else {
                    "not found"
                },
                if binary_available(&app.cfg.vllm) {
                    GOOD
                } else {
                    WARN
                },
            ));
        }
        SettingsRow::ModelsDir => {
            lines.push(Line::from(Span::styled(
                "Currently searched:",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )));
            for path in app.searched.iter().take(8) {
                lines.push(Line::from(Span::styled(
                    format!("  - {}", path.display()),
                    Style::default().fg(MUTED),
                )));
            }
            let primary = primary_models_dir(&app.cfg);
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Opens with o:  ", Style::default().fg(MUTED)),
                Span::styled(
                    primary.to_string_lossy().to_string(),
                    Style::default().fg(TEXT),
                ),
            ]));
        }
        SettingsRow::LlamaBin | SettingsRow::VllmBin => {
            lines.push(Line::from(Span::styled(
                "Leave empty to use the default command name.",
                Style::default().fg(MUTED),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Examples",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(Span::styled(
                "  llama-server",
                Style::default().fg(TEXT),
            )));
            lines.push(Line::from(Span::styled(
                "  /opt/homebrew/bin/llama-server",
                Style::default().fg(TEXT),
            )));
            lines.push(Line::from(Span::styled(
                "  .venv/bin/vllm",
                Style::default().fg(TEXT),
            )));
        }
        SettingsRow::Rescan => {
            lines.push(kv("Models now", &app.models.len().to_string(), ACCENT));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Press Enter or r to scan again.",
                Style::default().fg(MUTED),
            )));
        }
        SettingsRow::OpenConfig | SettingsRow::OpenModels => {}
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Press Esc to go back. Settings are saved automatically.",
        Style::default().fg(MUTED),
    )));

    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_running(frame: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::vertical([Constraint::Length(9), Constraint::Min(5)]).split(area);

    let info = app.server.info().cloned();
    let (label, color) = match app.server.status() {
        ServerStatus::Idle => ("IDLE", MUTED),
        ServerStatus::Starting => ("LOADING", WARN),
        ServerStatus::Ready => ("READY", GOOD),
        ServerStatus::Exited(_) => ("STOPPED", BAD),
    };

    let block = panel_block(" Server ", color);
    let inner = block.inner(chunks[0]);
    frame.render_widget(block, chunks[0]);

    let mut lines = Vec::new();
    let model_name = info
        .as_ref()
        .map(|info| info.model_name.clone())
        .unwrap_or_else(|| "No model".to_string());
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {label} "),
            Style::default()
                .fg(Color::Black)
                .bg(color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            model_name,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    if let Some(endpoint) = app.server.endpoint() {
        let page_label = match info.as_ref().map(|info| info.backend) {
            Some(crate::config::Backend::Vllm) => "API page   ",
            _ => "Chat page  ",
        };
        lines.push(Line::from(vec![
            Span::styled(page_label, Style::default().fg(MUTED)),
            Span::styled(endpoint.clone(), Style::default().fg(ACCENT)),
            Span::styled("   (o to open)", Style::default().fg(MUTED)),
        ]));
        if let Some(api) = app.server.api_endpoint() {
            lines.push(Line::from(vec![
                Span::styled("API        ", Style::default().fg(MUTED)),
                Span::styled(api, Style::default().fg(TEXT)),
            ]));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "Waiting for the server address...",
            Style::default().fg(MUTED),
        )));
    }

    if let Some(info) = &info {
        lines.push(Line::from(vec![
            Span::styled("Backend    ", Style::default().fg(MUTED)),
            Span::styled(info.backend.label().to_string(), Style::default().fg(TEXT)),
            Span::styled("    Settings    ", Style::default().fg(MUTED)),
            Span::styled(info.context.clone(), Style::default().fg(TEXT)),
        ]));
    }
    let uptime = if app.server.running() {
        app.server
            .uptime()
            .map(format_uptime)
            .unwrap_or_else(|| "-".to_string())
    } else {
        "-".to_string()
    };
    lines.push(Line::from(vec![
        Span::styled("Uptime     ", Style::default().fg(MUTED)),
        Span::styled(uptime, Style::default().fg(TEXT)),
        Span::styled("    Log    ", Style::default().fg(MUTED)),
        Span::styled(
            app.server
                .log_path()
                .and_then(|p| p.file_name())
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "-".to_string()),
            Style::default().fg(MUTED),
        ),
    ]));

    match app.server.status() {
        ServerStatus::Starting => lines.push(Line::from(Span::styled(
            "Loading the model... the first run can take a while. The log below shows progress.",
            Style::default().fg(WARN),
        ))),
        ServerStatus::Ready => {
            let message = match info.map(|info| info.backend) {
                Some(crate::config::Backend::Vllm) => {
                    "The server is ready. Press o to open the API docs."
                }
                _ => "The server is ready. Press o to open the chat page in your browser.",
            };
            lines.push(Line::from(Span::styled(message, Style::default().fg(GOOD))));
        }
        ServerStatus::Exited(message) => lines.push(Line::from(vec![
            Span::styled(message.clone(), Style::default().fg(BAD)),
            Span::styled(
                "  Press x to clear, then b to go back.",
                Style::default().fg(MUTED),
            ),
        ])),
        ServerStatus::Idle => lines.push(Line::from(Span::styled(
            "No server has been started yet.",
            Style::default().fg(MUTED),
        ))),
    }
    lines.push(Line::from(vec![
        Span::styled(
            "o",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            if app.cfg.backend == crate::config::Backend::Vllm {
                " open API   "
            } else {
                " open chat   "
            },
            Style::default().fg(TEXT),
        ),
        Span::styled(
            "x",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" stop/clear   ", Style::default().fg(TEXT)),
        Span::styled(
            "b",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" back   ", Style::default().fg(TEXT)),
        Span::styled(
            "↑/↓",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" scroll log", Style::default().fg(TEXT)),
    ]));

    frame.render_widget(Paragraph::new(Text::from(lines)), inner);

    draw_logs(frame, app, chunks[1]);
}

fn draw_logs(frame: &mut Frame, app: &mut App, area: Rect) {
    let title = if app.log_scroll > 0 {
        " Server log (scrolled up) "
    } else {
        " Server log "
    };
    let block = panel_block(title, MUTED);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let logs = app.server.logs();
    let height = inner.height as usize;
    if logs.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Waiting for output...",
                Style::default().fg(MUTED),
            ))
            .wrap(Wrap { trim: false }),
            inner,
        );
        return;
    }
    let end = logs.len().saturating_sub(app.log_scroll);
    let start = end.saturating_sub(height);
    let lines: Vec<Line> = logs[start..end]
        .iter()
        .map(|line| {
            let lower = line.to_lowercase();
            let style = if line.starts_with('$') {
                Style::default().fg(ACCENT)
            } else if lower.contains("error") || lower.contains("failed") {
                Style::default().fg(BAD)
            } else if lower.contains("warn") {
                Style::default().fg(WARN)
            } else {
                Style::default().fg(TEXT)
            };
            Line::from(Span::styled(line.clone(), style))
        })
        .collect();
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let lines = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(area);
    let hints: Vec<(&str, &str)> = match app.screen {
        Screen::Models if area.width < 88 => vec![
            ("↑/↓", "move"),
            ("Enter", "configure"),
            ("L", "launch"),
            ("s", "settings"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Screen::Models => vec![
            ("↑/↓", "move"),
            ("Enter", "configure"),
            ("L", "launch"),
            ("s", "settings"),
            ("r", "rescan"),
            ("o", "folder"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Screen::Configure => vec![
            ("←/→", "change"),
            ("Enter", "edit"),
            ("Tab", "section"),
            ("L", "launch"),
            ("c", "command"),
            ("Esc", "back"),
        ],
        Screen::Running => vec![
            (
                "o",
                if app.cfg.backend == crate::config::Backend::Vllm {
                    "open API"
                } else {
                    "open chat"
                },
            ),
            ("x", "stop/clear"),
            ("↑/↓", "logs"),
            ("b", "back"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Screen::Settings => vec![
            ("↑/↓", "move"),
            ("←/→", "backend"),
            ("Enter", "edit/run"),
            ("r", "rescan"),
            ("Esc", "back"),
        ],
    };
    let mut spans = Vec::new();
    for (key, label) in hints {
        spans.push(Span::styled(
            format!(" {key} "),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("{label}  "),
            Style::default().fg(TEXT),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), lines[0]);

    if let Some(toast) = &app.toast {
        let color = match toast.tone {
            Tone::Info => ACCENT,
            Tone::Good => GOOD,
            Tone::Warn => WARN,
            Tone::Error => BAD,
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" {} ", toast.message),
                Style::default().fg(color),
            ))),
            lines[1],
        );
    } else {
        let tip = match app.screen {
            Screen::Models => {
                "Tip: press Enter to tune settings, or L to run with the saved settings."
            }
            Screen::Configure => {
                "Tip: the panel on the right explains every option in plain language."
            }
            Screen::Running => "Tip: the server keeps running while you browse other models.",
            Screen::Settings => {
                "Tip: Ollama, LM Studio, ~/models, and .cache/llama.cpp are always searched."
            }
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(tip, Style::default().fg(MUTED)))),
            lines[1],
        );
    }
}

fn draw_help(frame: &mut Frame, app: &App) {
    let area = centered_rect(frame.area(), 84, 92);
    frame.render_widget(Clear, area);
    let block = panel_block(" Help ", ACCENT);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines: Vec<Line> = app
        .help_text()
        .into_iter()
        .map(|line| {
            if line.is_empty() {
                Line::from("")
            } else if !line.starts_with(' ') {
                Line::from(Span::styled(
                    line,
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(Span::styled(line, Style::default().fg(TEXT)))
            }
        })
        .collect();
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_modal(frame: &mut Frame, app: &App) {
    let Some(modal) = &app.modal else {
        return;
    };
    let (title, body, tone, hint) = match modal {
        Modal::Message { title, body, tone } => {
            (title.clone(), body.clone(), *tone, "Enter or Esc to close")
        }
        Modal::ConfirmLaunch => {
            let current = app
                .server
                .info()
                .map(|info| info.model_name.clone())
                .unwrap_or_else(|| "another model".to_string());
            (
                "Replace the running server?".to_string(),
                vec![
                    format!("A server for {current} is already running."),
                    String::new(),
                    "Starting a new model will stop it first.".to_string(),
                    "Your current chat page will disconnect.".to_string(),
                ],
                Tone::Warn,
                "Enter to replace    Esc to cancel",
            )
        }
        Modal::ConfirmQuit => (
            "Quit llm?".to_string(),
            vec![
                "The local model server is still running.".to_string(),
                String::new(),
                "Quitting will stop it. Your chat page will disconnect.".to_string(),
            ],
            Tone::Warn,
            "Enter or y to quit    Esc to stay",
        ),
    };

    let height = (body.len() as u16 + 7)
        .min(frame.area().height.saturating_sub(4))
        .max(7);
    let area = centered_rect_fixed(frame.area(), 66, height);
    frame.render_widget(Clear, area);
    let color = match tone {
        Tone::Info => ACCENT,
        Tone::Good => GOOD,
        Tone::Warn => WARN,
        Tone::Error => BAD,
    };
    let block = panel_block(&format!(" {title} "), color);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = body
        .into_iter()
        .map(|line| Line::from(Span::styled(line, Style::default().fg(TEXT))))
        .collect();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        hint,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )));
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_editor(frame: &mut Frame, app: &App) {
    let Some(edit) = &app.editing else {
        return;
    };
    let area = centered_rect_fixed(
        frame.area(),
        70,
        8.min(frame.area().height.saturating_sub(2)).max(6),
    );
    frame.render_widget(Clear, area);
    let block = panel_block(&format!(" Edit: {} ", app.edit_label()), ACCENT);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = vec![
        Line::from(Span::styled(
            edit.buffer.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Type a value. Enter saves, Esc cancels.",
            Style::default().fg(MUTED),
        )),
        Line::from(Span::styled(
            "Tip: paths can start with ~/ and numbers can be typed exactly.",
            Style::default().fg(MUTED),
        )),
    ];
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn panel_block(title: &str, color: Color) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(color))
        .style(Style::default().bg(PANEL_BG))
        .title(Line::from(Span::styled(
            title.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )))
}

fn kv(label: &str, value: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<14}"), Style::default().fg(MUTED)),
        Span::styled(value.to_string(), Style::default().fg(color)),
    ])
}

fn centered_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vertical = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(area);
    let horizontal = Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(vertical[1]);
    horizontal[1]
}

fn centered_rect_fixed(area: Rect, percent_x: u16, height: u16) -> Rect {
    let width = area.width.saturating_mul(percent_x) / 100;
    let width = width.max(30).min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}
