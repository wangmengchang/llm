use crate::backend::{
    binary_available, build_plan, find_free_port, open_path, open_url, port_open,
};
use crate::config::{primary_models_dir, Backend, Config, Settings, SpecType};
use crate::fields::{fields_for, Field, FieldKind, TABS};
use crate::models::{self, expand_tilde, Model};
use crate::server::{RunInfo, ServerManager};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use ratatui::widgets::ListState;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Models,
    Configure,
    Running,
    Settings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Info,
    Good,
    Warn,
    Error,
}

#[derive(Clone, Debug)]
pub struct Toast {
    pub message: String,
    pub tone: Tone,
    pub at: Instant,
}

#[derive(Clone, Debug)]
pub enum Modal {
    Message {
        title: String,
        body: Vec<String>,
        tone: Tone,
    },
    ConfirmLaunch,
    ConfirmQuit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditTarget {
    Field(Field),
    ModelsDir,
    LlamaBin,
    VllmBin,
}

#[derive(Clone, Debug)]
pub struct EditState {
    pub target: EditTarget,
    pub buffer: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UiAreas {
    pub body_height: u16,
    pub models_list: Option<Rect>,
    pub cfg_list: Option<Rect>,
    pub settings_list: Option<Rect>,
    pub cfg_tabs: Option<Rect>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsRow {
    Backend,
    ModelsDir,
    LlamaBin,
    VllmBin,
    Rescan,
    OpenConfig,
    OpenModels,
}

pub const SETTINGS_ROWS: [SettingsRow; 7] = [
    SettingsRow::Backend,
    SettingsRow::ModelsDir,
    SettingsRow::LlamaBin,
    SettingsRow::VllmBin,
    SettingsRow::Rescan,
    SettingsRow::OpenConfig,
    SettingsRow::OpenModels,
];

impl SettingsRow {
    pub fn label(self) -> &'static str {
        match self {
            SettingsRow::Backend => "Backend",
            SettingsRow::ModelsDir => "Extra models folder",
            SettingsRow::LlamaBin => "llama-server path",
            SettingsRow::VllmBin => "vLLM path",
            SettingsRow::Rescan => "Rescan models",
            SettingsRow::OpenConfig => "Open config folder",
            SettingsRow::OpenModels => "Open models folder",
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            SettingsRow::Backend => {
                "llama.cpp is the easiest way to run GGUF files and works on CPU or GPU. \
                 vLLM is faster for serving Hugging Face model folders on NVIDIA GPUs."
            }
            SettingsRow::ModelsDir => {
                "llm already searches the standard Ollama, LM Studio, ~/models, and \
                 .cache/llama.cpp folders. Add another folder here only if you keep models \
                 somewhere else. Leave it empty to use the standard folders."
            }
            SettingsRow::LlamaBin => {
                "The llama-server program to run. If it is on your PATH, just the name \
                 `llama-server` is enough. Otherwise enter the full path, for example \
                 /opt/homebrew/bin/llama-server."
            }
            SettingsRow::VllmBin => {
                "The vllm command to run. If it is inside a Python virtual environment, enter \
                 the full path to that environment's vllm executable."
            }
            SettingsRow::Rescan => {
                "Looks for models again in every standard folder plus your extra folder. \
                 You can also press r on the model screen."
            }
            SettingsRow::OpenConfig => {
                "Opens the folder that holds config.toml and the server logs."
            }
            SettingsRow::OpenModels => {
                "Opens your models folder in the file manager so you can add downloaded models."
            }
        }
    }
}

pub struct App {
    pub cfg: Config,
    pub models: Vec<Model>,
    pub searched: Vec<PathBuf>,
    pub scan_notes: Vec<String>,
    pub screen: Screen,
    pub models_state: ListState,
    pub cfg_tab: usize,
    pub cfg_state: ListState,
    pub settings_state: ListState,
    pub config_settings: Settings,
    pub editing: Option<EditState>,
    pub show_command: bool,
    pub show_help: bool,
    pub modal: Option<Modal>,
    pub server: ServerManager,
    pub toast: Option<Toast>,
    pub quit: bool,
    pub ui: UiAreas,
    pub log_scroll: usize,
    ready_seen: bool,
}

impl App {
    pub fn new(cfg: Config) -> Self {
        let mut app = Self {
            config_settings: cfg.settings.clone(),
            cfg,
            models: Vec::new(),
            searched: Vec::new(),
            scan_notes: Vec::new(),
            screen: Screen::Models,
            models_state: ListState::default(),
            cfg_tab: 0,
            cfg_state: ListState::default(),
            settings_state: ListState::default(),
            editing: None,
            show_command: false,
            show_help: false,
            modal: None,
            server: ServerManager::new(),
            toast: None,
            quit: false,
            ui: UiAreas::default(),
            log_scroll: 0,
            ready_seen: false,
        };
        app.rescan(false);
        app
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.models_state
            .selected()
            .filter(|idx| *idx < self.models.len())
            .or_else(|| (!self.models.is_empty()).then_some(0))
    }

    pub fn selected_model(&self) -> Option<&Model> {
        self.selected_index().and_then(|idx| self.models.get(idx))
    }

    pub fn notify(&mut self, message: impl Into<String>, tone: Tone) {
        self.toast = Some(Toast {
            message: message.into(),
            tone,
            at: Instant::now(),
        });
    }

    pub fn rescan(&mut self, announce: bool) {
        let previous = self.selected_model().map(|model| model.path.clone());
        let result = models::scan(self.cfg.backend, &self.cfg);
        self.models = result.models;
        self.searched = result.searched;
        self.scan_notes = result.notes;

        let index = previous
            .and_then(|path| self.models.iter().position(|model| model.path == path))
            .unwrap_or(0);
        if self.models.is_empty() {
            self.models_state.select(None);
        } else {
            self.models_state
                .select(Some(index.min(self.models.len() - 1)));
        }
        self.cfg_state.select(Some(0));
        self.cfg_tab = self.cfg_tab.min(TABS.len() - 1);
        self.settings_state
            .select(Some(self.settings_state.selected().unwrap_or(0)));

        if announce {
            let tone = if self.models.is_empty() {
                Tone::Warn
            } else {
                Tone::Good
            };
            let skipped = self.scan_notes.len();
            self.notify(
                if skipped > 0 {
                    format!(
                        "Found {} for {}. {} file(s) skipped.",
                        crate::models::count_label(self.models.len()),
                        self.cfg.backend.label(),
                        skipped
                    )
                } else {
                    format!(
                        "Found {} for {}.",
                        crate::models::count_label(self.models.len()),
                        self.cfg.backend.label()
                    )
                },
                tone,
            );
        }
    }

    pub fn move_models(&mut self, delta: i64) {
        let len = self.models.len();
        if len == 0 {
            return;
        }
        let current = self.selected_index().unwrap_or(0) as i64;
        let next = (current + delta).clamp(0, len as i64 - 1) as usize;
        self.models_state.select(Some(next));
    }

    pub fn set_model(&mut self, index: usize) {
        if index < self.models.len() {
            self.models_state.select(Some(index));
        }
    }

    pub fn open_configure(&mut self) {
        let Some(model) = self.selected_model() else {
            return;
        };
        let path = model.path.clone();
        self.config_settings = self.cfg.settings_for(&path);
        self.cfg_tab = 0;
        self.cfg_state.select(Some(0));
        self.editing = None;
        self.show_command = false;
        self.screen = Screen::Configure;
    }

    pub fn open_settings(&mut self) {
        self.settings_state.select(Some(0));
        self.screen = Screen::Settings;
    }

    pub fn leave_configure(&mut self) {
        if let Some(model) = self.selected_model() {
            let path = model.path.clone();
            self.cfg.store_settings(&path, &self.config_settings);
            self.cfg.settings = self.config_settings.clone();
        }
        self.editing = None;
        self.show_command = false;
        self.screen = Screen::Models;
        self.save_config();
    }

    pub fn save_config(&mut self) {
        if let Err(err) = self.cfg.save() {
            self.notify(format!("Could not save settings: {err}"), Tone::Error);
        }
    }

    pub fn request_launch(&mut self) {
        if self.selected_model().is_none() {
            self.notify("Pick a model first.", Tone::Warn);
            return;
        }
        if self.screen != Screen::Configure {
            if let Some(model) = self.selected_model() {
                self.config_settings = self.cfg.settings_for(&model.path);
            }
        }
        if self.server.running() {
            self.modal = Some(Modal::ConfirmLaunch);
            return;
        }
        self.launch_selected();
    }

    fn launch_selected(&mut self) {
        let Some(model) = self.selected_model().cloned() else {
            return;
        };
        let backend = self.cfg.backend;
        let mut settings = self.config_settings.clone();
        settings.sanitize();

        let program = self.cfg.binary(backend).to_string();
        if !binary_available(&program) {
            self.modal = Some(Modal::Message {
                title: format!("{} was not found", backend.label()),
                body: wrap_lines(&format!(
                    "llm could not find `{program}`.\n\n{}",
                    backend.install_hint()
                )),
                tone: Tone::Warn,
            });
            return;
        }

        let requested_port = settings.port;
        let chosen_port = if port_open(requested_port) {
            find_free_port(requested_port.saturating_add(1))
        } else {
            requested_port
        };
        settings.port = chosen_port;

        let plan = match build_plan(backend, &settings, &model.path, &program) {
            Ok(plan) => plan,
            Err(err) => {
                self.modal = Some(Modal::Message {
                    title: "Almost there".to_string(),
                    body: wrap_lines(&err.to_string()),
                    tone: Tone::Warn,
                });
                return;
            }
        };

        let info = RunInfo {
            model_name: model.name.clone(),
            backend,
            host: settings.host.clone(),
            port: chosen_port,
            context: settings.summary(backend),
        };
        match self.server.start(&plan, info, &Config::log_dir()) {
            Ok(()) => {
                if let Some(model_idx) = self.selected_index() {
                    let path = self.models[model_idx].path.clone();
                    self.cfg.store_settings(&path, &settings);
                    self.cfg.settings = settings.clone();
                }
                self.config_settings = settings;
                self.log_scroll = 0;
                self.ready_seen = false;
                self.screen = Screen::Running;
                self.save_config();
                if chosen_port != requested_port {
                    self.notify(
                        format!("Port {requested_port} was busy, so llm started on {chosen_port}."),
                        Tone::Info,
                    );
                } else {
                    self.notify(
                        format!("Starting {}... this can take a moment.", model.name),
                        Tone::Good,
                    );
                }
            }
            Err(err) => {
                self.modal = Some(Modal::Message {
                    title: "Could not start the server".to_string(),
                    body: wrap_lines(&err.to_string()),
                    tone: Tone::Error,
                });
            }
        }
    }

    pub fn request_quit(&mut self) {
        if self.server.running() {
            self.modal = Some(Modal::ConfirmQuit);
        } else {
            self.quit = true;
        }
    }

    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        if let Some(path) = self.selected_model().map(|model| model.path.clone()) {
            if self.screen == Screen::Configure {
                self.cfg.store_settings(&path, &self.config_settings);
            }
        }
        self.server.stop();
        self.cfg.save()
    }

    pub fn on_tick(&mut self) {
        if let Some(message) = self.server.poll() {
            self.notify(message, Tone::Warn);
        }
        let ready = matches!(self.server.status(), crate::server::ServerStatus::Ready);
        if ready && !self.ready_seen {
            self.ready_seen = true;
            let page = if self.cfg.backend == Backend::Vllm {
                "API docs"
            } else {
                "chat page"
            };
            let message = match self.server.info().map(|info| info.port) {
                Some(port) => {
                    format!("Server is ready on port {port}. Press o to open the {page}.")
                }
                None => format!("Server is ready. Press o to open the {page}."),
            };
            self.notify(message, Tone::Good);
        } else if !ready {
            self.ready_seen = false;
        }
        if let Some(toast) = &self.toast {
            if toast.at.elapsed() > Duration::from_secs(8) {
                self.toast = None;
            }
        }
        let max_scroll = self.server.logs().len();
        self.log_scroll = self.log_scroll.min(max_scroll);
    }

    pub fn fields(&self) -> &'static [Field] {
        fields_for(self.cfg.backend, self.cfg_tab)
    }

    pub fn cfg_row(&self) -> usize {
        self.cfg_state
            .selected()
            .unwrap_or(0)
            .min(self.fields().len().saturating_sub(1))
    }

    pub fn current_field(&self) -> Option<Field> {
        self.fields().get(self.cfg_row()).copied()
    }

    pub fn set_tab(&mut self, tab: usize) {
        self.cfg_tab = tab.min(TABS.len() - 1);
        self.cfg_state.select(Some(0));
    }

    pub fn move_cfg(&mut self, delta: i64) {
        let len = self.fields().len();
        move_list_state(&mut self.cfg_state, len, delta);
    }

    pub fn move_settings(&mut self, delta: i64) {
        move_list_state(&mut self.settings_state, SETTINGS_ROWS.len(), delta);
    }

    pub fn settings_row(&self) -> SettingsRow {
        SETTINGS_ROWS[self
            .settings_state
            .selected()
            .unwrap_or(0)
            .min(SETTINGS_ROWS.len() - 1)]
    }

    pub fn settings_value(&self, row: SettingsRow) -> String {
        match row {
            SettingsRow::Backend => self.cfg.backend.label().to_string(),
            SettingsRow::ModelsDir => self
                .cfg
                .models_dir
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "Auto".to_string()),
            SettingsRow::LlamaBin => format!(
                "{}  {}",
                self.cfg.llama_server,
                if binary_available(&self.cfg.llama_server) {
                    "found"
                } else {
                    "missing"
                }
            ),
            SettingsRow::VllmBin => format!(
                "{}  {}",
                self.cfg.vllm,
                if binary_available(&self.cfg.vllm) {
                    "found"
                } else {
                    "missing"
                }
            ),
            SettingsRow::Rescan => "Press Enter".to_string(),
            SettingsRow::OpenConfig => "Press Enter".to_string(),
            SettingsRow::OpenModels => "Press Enter".to_string(),
        }
    }

    pub fn field_value(&self, field: Field) -> String {
        let s = &self.config_settings;
        match field {
            Field::Preset => s.preset.label().to_string(),
            Field::Context => {
                if s.context_size == 0 {
                    "Auto (default)".to_string()
                } else {
                    format!(
                        "{} tokens",
                        crate::config::format_token_count(s.context_size as u64)
                    )
                }
            }
            Field::MaxTokens => {
                if s.max_tokens < 0 {
                    "Auto".to_string()
                } else {
                    s.max_tokens.to_string()
                }
            }
            Field::Temperature => format!("{:.2}", s.temperature),
            Field::TopP => format!("{:.2}", s.top_p),
            Field::RepeatPenalty => format!("{:.2}", s.repeat_penalty),
            Field::Effort => s.effort.label().to_string(),
            Field::ReasoningBudget => {
                if s.reasoning_budget < 0 {
                    "Unlimited".to_string()
                } else if s.reasoning_budget == 0 {
                    "Off (no thinking)".to_string()
                } else {
                    format!("{} tokens", s.reasoning_budget)
                }
            }
            Field::GpuLayers => match s.gpu_layers {
                -1 => "Auto".to_string(),
                -2 => "All layers".to_string(),
                0 => "0 (CPU only)".to_string(),
                n => format!("{n} layers"),
            },
            Field::Batch => s.batch_size.to_string(),
            Field::Ubatch => s.ubatch_size.to_string(),
            Field::Threads => {
                if s.threads == 0 {
                    "Auto".to_string()
                } else {
                    s.threads.to_string()
                }
            }
            Field::Parallel => s.parallel.to_string(),
            Field::FlashAttn => s.flash_attn.label().to_string(),
            Field::Jinja => on_off(s.jinja),
            Field::Mlock => on_off(s.mlock),
            Field::Seed => {
                if s.seed < 0 {
                    "Auto (random)".to_string()
                } else {
                    s.seed.to_string()
                }
            }
            Field::SpecType => s.spec_type.label().to_string(),
            Field::DraftModel => {
                if s.draft_model.trim().is_empty() {
                    "None".to_string()
                } else {
                    Path::new(s.draft_model.trim())
                        .file_name()
                        .map(|name| name.to_string_lossy().to_string())
                        .unwrap_or_else(|| s.draft_model.clone())
                }
            }
            Field::DraftMax => s.draft_max.to_string(),
            Field::DraftMin => s.draft_min.to_string(),
            Field::Host => s.host.clone(),
            Field::Port => s.port.to_string(),
            Field::ExtraArgs => {
                if s.extra_args.trim().is_empty() {
                    "None".to_string()
                } else {
                    s.extra_args.clone()
                }
            }
            Field::VllmGpuMem => format!("{:.0}%", s.vllm_gpu_mem * 100.0),
            Field::VllmMaxSeqs => s.vllm_max_seqs.to_string(),
            Field::VllmBatchTokens => s.vllm_batch_tokens.to_string(),
        }
    }

    pub fn adjust_field(&mut self, field: Field, delta: i32) {
        if field == Field::Preset {
            self.config_settings.cycle_preset(delta);
            return;
        }
        if field == Field::DraftModel {
            self.cycle_draft_model(delta);
            return;
        }
        let backend = self.cfg.backend;
        let s = &mut self.config_settings;
        match field {
            Field::Preset => {}
            Field::Context => {
                s.context_size = step_u32(s.context_size, delta, 1024, 0, 1_048_576);
            }
            Field::MaxTokens => {
                s.max_tokens = step_auto(s.max_tokens, delta, 256, 256);
            }
            Field::Temperature => {
                s.temperature = step_f32(s.temperature, delta, 0.05, 0.0, 2.0);
            }
            Field::TopP => {
                s.top_p = step_f32(s.top_p, delta, 0.05, 0.05, 1.0);
            }
            Field::RepeatPenalty => {
                s.repeat_penalty = step_f32(s.repeat_penalty, delta, 0.05, 0.5, 2.0);
            }
            Field::Effort => s.effort = s.effort.next(delta),
            Field::ReasoningBudget => {
                s.reasoning_budget = step_auto(s.reasoning_budget, delta, 0, 256);
            }
            Field::GpuLayers => {
                s.gpu_layers = step_gpu_layers(s.gpu_layers, delta);
            }
            Field::Batch => {
                s.batch_size = step_u32(s.batch_size, delta, 256, 32, 65_536);
            }
            Field::Ubatch => {
                s.ubatch_size = step_u32(s.ubatch_size, delta, 128, 32, 65_536);
            }
            Field::Threads => {
                s.threads = step_u16(s.threads, delta, 1, 0, 256);
            }
            Field::Parallel => {
                s.parallel = step_u16(s.parallel, delta, 1, 1, 64);
            }
            Field::FlashAttn => s.flash_attn = s.flash_attn.next(delta),
            Field::Jinja => s.jinja = !s.jinja,
            Field::Mlock => s.mlock = !s.mlock,
            Field::Seed => {
                s.seed = step_auto(s.seed, delta, 0, 1);
            }
            Field::SpecType => {
                s.spec_type = s.spec_type.next(delta, backend);
                if s.spec_type == SpecType::Off && backend == Backend::Vllm {
                    s.draft_model.clear();
                }
            }
            Field::DraftModel => {}
            Field::DraftMax => {
                s.draft_max = step_u32(s.draft_max, delta, 1, 0, 16);
                s.draft_min = s.draft_min.min(s.draft_max);
            }
            Field::DraftMin => {
                s.draft_min = step_u32(s.draft_min, delta, 1, 0, s.draft_max);
            }
            Field::Host | Field::ExtraArgs => return,
            Field::Port => {
                s.port = step_u16(s.port, delta, 1, 1, 65_535);
            }
            Field::VllmGpuMem => {
                s.vllm_gpu_mem = step_f32(s.vllm_gpu_mem, delta, 0.05, 0.1, 0.99);
            }
            Field::VllmMaxSeqs => {
                s.vllm_max_seqs = step_u16(s.vllm_max_seqs, delta, 1, 1, 4096);
            }
            Field::VllmBatchTokens => {
                s.vllm_batch_tokens = step_u32(s.vllm_batch_tokens, delta, 256, 256, 1_048_576);
            }
        }
        s.sanitize();
        s.mark_custom();
    }

    fn cycle_draft_model(&mut self, delta: i32) {
        let current_model = self.selected_model().map(|m| m.path.clone());
        let candidates: Vec<(String, String)> = self
            .models
            .iter()
            .filter(|m| Some(&m.path) != current_model.as_ref())
            .map(|m| (m.name.clone(), m.path.to_string_lossy().to_string()))
            .collect();
        if candidates.is_empty() {
            self.notify(
                "No other GGUF models found to use as a draft model.",
                Tone::Warn,
            );
            return;
        }
        let current = self.config_settings.draft_model.trim().to_string();
        let idx = candidates
            .iter()
            .position(|(_, path)| *path == current)
            .map(|i| i as i32)
            .unwrap_or(-1);
        let len = candidates.len() as i32 + 1;
        let next = (idx + delta).rem_euclid(len) - 1;
        if next < 0 {
            self.config_settings.draft_model.clear();
            if self.config_settings.spec_type == SpecType::Draft {
                self.config_settings.spec_type = SpecType::Off;
            }
        } else {
            self.config_settings.draft_model = candidates[next as usize].1.clone();
            if self.config_settings.spec_type == SpecType::Off {
                self.config_settings.spec_type = SpecType::Draft;
            }
        }
        self.config_settings.mark_custom();
    }

    pub fn enter_field(&mut self) {
        let Some(field) = self.current_field() else {
            return;
        };
        match field.kind() {
            FieldKind::Bool | FieldKind::Cycle | FieldKind::Preset => {
                self.adjust_field(field, 1);
            }
            FieldKind::Text | FieldKind::Integer | FieldKind::Float | FieldKind::GpuLayers => {
                if let Some(value) = self.raw_field_value(field) {
                    self.begin_edit(EditTarget::Field(field), value);
                }
            }
        }
    }

    fn raw_field_value(&self, field: Field) -> Option<String> {
        let s = &self.config_settings;
        Some(match field {
            Field::Context => {
                if s.context_size == 0 {
                    String::new()
                } else {
                    s.context_size.to_string()
                }
            }
            Field::MaxTokens => {
                if s.max_tokens < 0 {
                    String::new()
                } else {
                    s.max_tokens.to_string()
                }
            }
            Field::Temperature => format!("{:.2}", s.temperature),
            Field::TopP => format!("{:.2}", s.top_p),
            Field::RepeatPenalty => format!("{:.2}", s.repeat_penalty),
            Field::ReasoningBudget => {
                if s.reasoning_budget < 0 {
                    String::new()
                } else {
                    s.reasoning_budget.to_string()
                }
            }
            Field::GpuLayers => match s.gpu_layers {
                -1 => String::new(),
                -2 => "all".to_string(),
                n => n.to_string(),
            },
            Field::Batch => s.batch_size.to_string(),
            Field::Ubatch => s.ubatch_size.to_string(),
            Field::Threads => {
                if s.threads == 0 {
                    String::new()
                } else {
                    s.threads.to_string()
                }
            }
            Field::Parallel => s.parallel.to_string(),
            Field::Seed => {
                if s.seed < 0 {
                    String::new()
                } else {
                    s.seed.to_string()
                }
            }
            Field::DraftMax => s.draft_max.to_string(),
            Field::DraftMin => s.draft_min.to_string(),
            Field::Host => s.host.clone(),
            Field::Port => s.port.to_string(),
            Field::ExtraArgs => s.extra_args.clone(),
            Field::VllmGpuMem => format!("{:.2}", s.vllm_gpu_mem),
            Field::VllmMaxSeqs => s.vllm_max_seqs.to_string(),
            Field::VllmBatchTokens => s.vllm_batch_tokens.to_string(),
            _ => return None,
        })
    }

    pub fn begin_edit(&mut self, target: EditTarget, buffer: String) {
        self.editing = Some(EditState { target, buffer });
    }

    pub fn cancel_edit(&mut self) {
        self.editing = None;
    }

    pub fn commit_edit(&mut self) {
        let Some(edit) = self.editing.clone() else {
            return;
        };
        let text = edit.buffer.trim().to_string();
        match edit.target {
            EditTarget::Field(field) => match self.apply_field_text(field, &text) {
                Ok(()) => {
                    self.config_settings.sanitize();
                    self.config_settings.mark_custom();
                    self.editing = None;
                }
                Err(message) => {
                    self.notify(message, Tone::Warn);
                    self.editing = Some(edit);
                }
            },
            EditTarget::ModelsDir => {
                let missing = if text.is_empty() {
                    self.cfg.models_dir = None;
                    None
                } else {
                    let path = expand_tilde(Path::new(&text));
                    let missing = (!path.exists()).then(|| path.clone());
                    self.cfg.models_dir = Some(path);
                    missing
                };
                self.editing = None;
                self.save_config();
                self.rescan(true);
                if let Some(path) = missing {
                    self.notify(
                        format!("That folder does not exist yet: {}", path.display()),
                        Tone::Warn,
                    );
                }
            }
            EditTarget::LlamaBin => {
                self.cfg.llama_server = if text.is_empty() {
                    "llama-server".to_string()
                } else {
                    text
                };
                self.editing = None;
                self.save_config();
                if !binary_available(&self.cfg.llama_server) {
                    self.notify(
                        format!(
                            "`{}` was not found. Install it or fix the path.",
                            self.cfg.llama_server
                        ),
                        Tone::Warn,
                    );
                }
            }
            EditTarget::VllmBin => {
                self.cfg.vllm = if text.is_empty() {
                    "vllm".to_string()
                } else {
                    text
                };
                self.editing = None;
                self.save_config();
                if !binary_available(&self.cfg.vllm) {
                    self.notify(
                        format!(
                            "`{}` was not found. Install it or fix the path.",
                            self.cfg.vllm
                        ),
                        Tone::Warn,
                    );
                }
            }
        }
    }

    fn apply_field_text(&mut self, field: Field, text: &str) -> Result<(), String> {
        let s = &mut self.config_settings;
        let auto = |text: &str| {
            text.is_empty()
                || text.eq_ignore_ascii_case("auto")
                || text.eq_ignore_ascii_case("default")
        };
        match field {
            Field::Context => {
                s.context_size = if auto(text) {
                    0
                } else {
                    text.parse::<u32>()
                        .map_err(|_| "Context must be a number, or \"auto\".".to_string())?
                };
            }
            Field::MaxTokens => {
                s.max_tokens = if auto(text) {
                    -1
                } else {
                    text.parse::<i64>().map_err(|_| {
                        "Max reply tokens must be a number, or \"auto\".".to_string()
                    })?
                };
            }
            Field::Temperature => {
                s.temperature = parse_f32(text, "Temperature")?;
            }
            Field::TopP => s.top_p = parse_f32(text, "Top-p")?,
            Field::RepeatPenalty => s.repeat_penalty = parse_f32(text, "Repeat penalty")?,
            Field::ReasoningBudget => {
                s.reasoning_budget = if auto(text) {
                    -1
                } else {
                    text.parse::<i64>().map_err(|_| {
                        "Reasoning budget must be a number, or \"auto\".".to_string()
                    })?
                };
            }
            Field::GpuLayers => {
                s.gpu_layers = if auto(text) {
                    -1
                } else if text.eq_ignore_ascii_case("all") {
                    -2
                } else {
                    text.parse::<i64>().map_err(|_| {
                        "GPU layers must be a number, \"auto\", or \"all\".".to_string()
                    })?
                };
            }
            Field::Batch => s.batch_size = parse_u32(text, "Batch size")?,
            Field::Ubatch => s.ubatch_size = parse_u32(text, "Micro-batch")?,
            Field::Threads => {
                s.threads = if auto(text) {
                    0
                } else {
                    parse_u16(text, "CPU threads")?
                }
            }
            Field::Parallel => s.parallel = parse_u16(text, "Parallel chats")?,
            Field::Seed => {
                s.seed = if auto(text) {
                    -1
                } else {
                    text.parse::<i64>()
                        .map_err(|_| "Seed must be a number, or \"auto\".".to_string())?
                };
            }
            Field::DraftMax => s.draft_max = parse_u32(text, "Draft tokens")?,
            Field::DraftMin => s.draft_min = parse_u32(text, "Min draft tokens")?,
            Field::Host => {
                if text.is_empty() {
                    return Err("Host cannot be empty. Try 127.0.0.1.".to_string());
                }
                s.host = text.to_string();
            }
            Field::Port => s.port = parse_u16(text, "Port")?,
            Field::ExtraArgs => s.extra_args = text.to_string(),
            Field::VllmGpuMem => {
                let value = if let Some(percent) = text.strip_suffix('%') {
                    percent
                        .parse::<f32>()
                        .map_err(|_| "GPU memory must be 0.90 or 90%.".to_string())?
                        / 100.0
                } else {
                    parse_f32(text, "GPU memory")?
                };
                s.vllm_gpu_mem = value;
            }
            Field::VllmMaxSeqs => s.vllm_max_seqs = parse_u16(text, "Max requests")?,
            Field::VllmBatchTokens => s.vllm_batch_tokens = parse_u32(text, "Batched tokens")?,
            _ => return Err("This setting is changed with Left/Right.".to_string()),
        }
        Ok(())
    }

    pub fn change_backend(&mut self) {
        if self.server.running() {
            self.notify(
                "Stop the running server before changing backends.",
                Tone::Warn,
            );
            return;
        }
        self.cfg.backend = self.cfg.backend.next();
        if self.cfg.backend == Backend::Vllm
            && !matches!(
                self.config_settings.spec_type,
                SpecType::Off | SpecType::Draft
            )
        {
            self.config_settings.spec_type = SpecType::Off;
        }
        self.save_config();
        self.rescan(true);
        self.notify(
            format!(
                "Backend: {}. {} available for it.",
                self.cfg.backend.label(),
                crate::models::count_label(self.models.len())
            ),
            if self.models.is_empty() {
                Tone::Warn
            } else {
                Tone::Good
            },
        );
    }

    pub fn activate_settings_row(&mut self) {
        match self.settings_row() {
            SettingsRow::Backend => self.change_backend(),
            SettingsRow::ModelsDir => {
                let value = self
                    .cfg
                    .models_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                self.begin_edit(EditTarget::ModelsDir, value);
            }
            SettingsRow::LlamaBin => {
                let value = self.cfg.llama_server.clone();
                self.begin_edit(EditTarget::LlamaBin, value);
            }
            SettingsRow::VllmBin => {
                let value = self.cfg.vllm.clone();
                self.begin_edit(EditTarget::VllmBin, value);
            }
            SettingsRow::Rescan => {
                self.rescan(true);
            }
            SettingsRow::OpenConfig => {
                let dir = Config::path()
                    .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                    .unwrap_or_else(Config::log_dir);
                if let Err(err) = std::fs::create_dir_all(&dir) {
                    self.notify(
                        format!("Could not create config folder: {err}"),
                        Tone::Error,
                    );
                    return;
                }
                if let Err(err) = open_path(&dir) {
                    self.notify(format!("Could not open folder: {err}"), Tone::Error);
                }
            }
            SettingsRow::OpenModels => self.open_models_folder(),
        }
    }

    pub fn open_models_folder(&mut self) {
        let dir = primary_models_dir(&self.cfg);
        if let Err(err) = std::fs::create_dir_all(&dir) {
            self.notify(
                format!("Could not create models folder {}: {err}", dir.display()),
                Tone::Error,
            );
            return;
        }
        if let Err(err) = open_path(&dir) {
            self.notify(format!("Could not open folder: {err}"), Tone::Error);
        }
    }

    pub fn open_download_page(&mut self) {
        let url = match self.cfg.backend {
            Backend::LlamaCpp => "https://huggingface.co/models?library=gguf",
            Backend::Vllm => "https://huggingface.co/models",
        };
        if let Err(err) = open_url(url) {
            self.notify(format!("Could not open the browser: {err}"), Tone::Error);
        }
    }

    pub fn open_endpoint(&mut self) {
        if let Some(endpoint) = self.server.endpoint() {
            let target = if self.cfg.backend == Backend::Vllm {
                format!("{}/docs", endpoint.trim_end_matches('/'))
            } else {
                endpoint
            };
            if let Err(err) = open_url(&target) {
                self.notify(format!("Could not open the browser: {err}"), Tone::Error);
            }
        } else {
            self.notify("The server is not running.", Tone::Warn);
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind == crossterm::event::KeyEventKind::Release {
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
            self.request_quit();
            return;
        }
        if self.editing.is_some() {
            self.handle_edit_key(key);
            return;
        }
        if self.show_help {
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q')
            ) {
                self.show_help = false;
            }
            return;
        }
        if self.modal.is_some() {
            self.handle_modal_key(key);
            return;
        }

        match self.screen {
            Screen::Models => self.keys_models(key),
            Screen::Configure => self.keys_configure(key),
            Screen::Running => self.keys_running(key),
            Screen::Settings => self.keys_settings(key),
        }
    }

    fn keys_models(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.move_models(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_models(1),
            KeyCode::PageUp => self.move_models(-10),
            KeyCode::PageDown => self.move_models(10),
            KeyCode::Home | KeyCode::Char('g') => self.move_models(-(i64::MAX / 2)),
            KeyCode::End | KeyCode::Char('G') => self.move_models(i64::MAX / 2),
            KeyCode::Enter => self.open_configure(),
            KeyCode::Char('l') | KeyCode::Char('L') => self.request_launch(),
            KeyCode::Char('s') => self.open_settings(),
            KeyCode::Char('r') => self.rescan(true),
            KeyCode::Char('o') => self.open_models_folder(),
            KeyCode::Char('d') => self.open_download_page(),
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Char('q') => self.request_quit(),
            _ => {}
        }
    }

    fn keys_configure(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.leave_configure(),
            KeyCode::Up | KeyCode::Char('k') => self.move_cfg(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_cfg(1),
            KeyCode::PageUp => self.move_cfg(-5),
            KeyCode::PageDown => self.move_cfg(5),
            KeyCode::Home | KeyCode::Char('g') => self.move_cfg(-(i64::MAX / 2)),
            KeyCode::End | KeyCode::Char('G') => self.move_cfg(i64::MAX / 2),
            KeyCode::Left => {
                if let Some(field) = self.current_field() {
                    self.adjust_field(field, -1);
                }
            }
            KeyCode::Right => {
                if let Some(field) = self.current_field() {
                    self.adjust_field(field, 1);
                }
            }
            KeyCode::Enter => self.enter_field(),
            KeyCode::Tab => {
                let next = if key.modifiers.contains(KeyModifiers::SHIFT) {
                    (self.cfg_tab + TABS.len() - 1) % TABS.len()
                } else {
                    (self.cfg_tab + 1) % TABS.len()
                };
                self.set_tab(next);
            }
            KeyCode::Char('1') => self.set_tab(0),
            KeyCode::Char('2') => self.set_tab(1),
            KeyCode::Char('3') => self.set_tab(2),
            KeyCode::Char('c') => {
                if !self.show_command && self.ui.body_height < 18 {
                    self.notify(
                        "Make the terminal taller to preview the command.",
                        Tone::Warn,
                    );
                } else {
                    self.show_command = !self.show_command;
                }
            }
            KeyCode::Char('l') | KeyCode::Char('L') => self.request_launch(),
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Char('q') => self.request_quit(),
            _ => {}
        }
    }

    fn keys_running(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('x') => {
                if self.server.running() {
                    self.server.stop();
                    self.notify("Server stopped.", Tone::Info);
                } else {
                    self.server.clear();
                    self.notify("Cleared the previous server session.", Tone::Info);
                }
            }
            KeyCode::Char('b') | KeyCode::Esc => {
                if !self.server.running() {
                    self.server.clear();
                }
                self.screen = Screen::Models;
            }
            KeyCode::Char('o') => self.open_endpoint(),
            KeyCode::Up | KeyCode::PageUp | KeyCode::Char('k') => {
                self.log_scroll = self.log_scroll.saturating_add(5);
            }
            KeyCode::Down | KeyCode::PageDown | KeyCode::Char('j') => {
                self.log_scroll = self.log_scroll.saturating_sub(5);
            }
            KeyCode::End | KeyCode::Char('G') => self.log_scroll = 0,
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Char('q') => self.request_quit(),
            _ => {}
        }
    }

    fn keys_settings(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.save_config();
                self.screen = Screen::Models;
            }
            KeyCode::Up | KeyCode::Char('k') => self.move_settings(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_settings(1),
            KeyCode::Left | KeyCode::Right => {
                if self.settings_row() == SettingsRow::Backend {
                    self.change_backend();
                }
            }
            KeyCode::Enter => self.activate_settings_row(),
            KeyCode::Char('r') => self.rescan(true),
            KeyCode::Char('o') => self.open_models_folder(),
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Char('q') => self.request_quit(),
            _ => {}
        }
    }

    fn handle_edit_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.cancel_edit(),
            KeyCode::Enter => self.commit_edit(),
            KeyCode::Backspace => {
                if let Some(edit) = self.editing.as_mut() {
                    edit.buffer.pop();
                }
            }
            KeyCode::Delete => {
                if let Some(edit) = self.editing.as_mut() {
                    edit.buffer.clear();
                }
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(edit) = self.editing.as_mut() {
                    edit.buffer.clear();
                }
            }
            KeyCode::Char(c) => {
                if let Some(edit) = self.editing.as_mut() {
                    if edit.buffer.len() < 1024 {
                        edit.buffer.push(c);
                    }
                }
            }
            _ => {}
        }
    }

    fn handle_modal_key(&mut self, key: KeyEvent) {
        let modal = self.modal.clone();
        match modal {
            Some(Modal::Message { .. }) => {
                if matches!(
                    key.code,
                    KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char(' ')
                ) {
                    self.modal = None;
                }
            }
            Some(Modal::ConfirmQuit) => match key.code {
                KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.modal = None;
                    self.quit = true;
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                    self.modal = None;
                }
                _ => {}
            },
            Some(Modal::ConfirmLaunch) => match key.code {
                KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.modal = None;
                    self.server.stop();
                    self.server.clear();
                    self.launch_selected();
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                    self.modal = None;
                }
                _ => {}
            },
            None => {}
        }
    }

    pub fn on_mouse(&mut self, mouse: MouseEvent) {
        if self.editing.is_some() || self.show_help || self.modal.is_some() {
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollUp => match self.screen {
                Screen::Models => self.move_models(-1),
                Screen::Configure => self.move_cfg(-1),
                Screen::Settings => self.move_settings(-1),
                Screen::Running => self.log_scroll = self.log_scroll.saturating_add(2),
            },
            MouseEventKind::ScrollDown => match self.screen {
                Screen::Models => self.move_models(1),
                Screen::Configure => self.move_cfg(1),
                Screen::Settings => self.move_settings(1),
                Screen::Running => self.log_scroll = self.log_scroll.saturating_sub(2),
            },
            MouseEventKind::Down(MouseButton::Left) => self.handle_click(mouse),
            _ => {}
        }
    }

    fn handle_click(&mut self, mouse: MouseEvent) {
        match self.screen {
            Screen::Models => {
                if let Some(rect) = self.ui.models_list {
                    if let Some(index) = row_at_steps(rect, self.models_state.offset(), mouse, 2) {
                        if index < self.models.len() {
                            self.set_model(index);
                        }
                    }
                }
            }
            Screen::Configure => {
                if let (Some(rect), Some(tabs)) = (self.ui.cfg_list, self.ui.cfg_tabs) {
                    if contains(tabs, mouse.column, mouse.row) {
                        let width = tabs.width.max(1);
                        let tab = ((mouse.column.saturating_sub(tabs.x)) as u32 * 3 / width as u32)
                            .min(2) as usize;
                        self.set_tab(tab);
                        return;
                    }
                    if let Some(index) = row_at(rect, self.cfg_state.offset(), mouse) {
                        if index < self.fields().len() {
                            self.cfg_state.select(Some(index));
                        }
                    }
                }
            }
            Screen::Settings => {
                if let Some(rect) = self.ui.settings_list {
                    if let Some(index) = row_at(rect, self.settings_state.offset(), mouse) {
                        if index < SETTINGS_ROWS.len() {
                            self.settings_state.select(Some(index));
                        }
                    }
                }
            }
            Screen::Running => {}
        }
    }

    pub fn edit_label(&self) -> &'static str {
        match self.editing.as_ref().map(|edit| edit.target) {
            Some(EditTarget::Field(field)) => field.label(),
            Some(EditTarget::ModelsDir) => "Extra models folder",
            Some(EditTarget::LlamaBin) => "llama-server path",
            Some(EditTarget::VllmBin) => "vLLM path",
            None => "",
        }
    }

    pub fn draft_candidates(&self) -> Vec<(String, PathBuf)> {
        let current = self.selected_model().map(|m| m.path.clone());
        self.models
            .iter()
            .filter(|m| Some(&m.path) != current.as_ref())
            .map(|m| (m.name.clone(), m.path.clone()))
            .collect()
    }

    pub fn help_text(&self) -> Vec<String> {
        vec![
            "Keys".to_string(),
            "  ↑/↓ move · ←/→ change · Enter edit · Tab section".to_string(),
            "  L launch · c command · r rescan · s settings · ? help · q quit".to_string(),
            "How it works".to_string(),
            "  llm starts llama.cpp or vLLM locally and opens a chat page.".to_string(),
            "  Nothing is uploaded; models and prompts stay on this computer.".to_string(),
            "Settings, in plain language".to_string(),
            "  Context       how much conversation the model remembers".to_string(),
            "  Max reply     longest answer".to_string(),
            "  Temperature   creativity; lower is more focused".to_string(),
            "  GPU layers    how much of the model goes on the graphics card".to_string(),
            "  Batch         how much text is processed at once".to_string(),
            "  Reasoning     thinking effort and thinking-token cap".to_string(),
            "  Speculation   draft model, MTP, DFlash, DSpark, N-gram".to_string(),
            "If something fails".to_string(),
            "  Out of memory: lower Context, set GPU layers to Auto.".to_string(),
            "  No server: press c to see the command, then check the log.".to_string(),
            "Press ? or Esc to close.".to_string(),
        ]
    }
}

fn move_list_state(state: &mut ListState, len: usize, delta: i64) {
    if len == 0 {
        state.select(None);
        return;
    }
    let current = state.selected().unwrap_or(0) as i64;
    let next = (current + delta).clamp(0, len as i64 - 1) as usize;
    state.select(Some(next));
}

fn on_off(value: bool) -> String {
    if value {
        "On".to_string()
    } else {
        "Off".to_string()
    }
}

fn step_u32(value: u32, delta: i32, step: u32, min: u32, max: u32) -> u32 {
    let step = step as i64 * delta as i64;
    (value as i64 + step).clamp(min as i64, max as i64) as u32
}

fn step_u16(value: u16, delta: i32, step: u16, min: u16, max: u16) -> u16 {
    let step = step as i64 * delta as i64;
    (value as i64 + step).clamp(min as i64, max as i64) as u16
}

fn step_f32(value: f32, delta: i32, step: f32, min: f32, max: f32) -> f32 {
    let next = value + step * delta as f32;
    (next.clamp(min, max) * 100.0).round() / 100.0
}

fn step_auto(value: i64, delta: i32, auto_step: i64, step: i64) -> i64 {
    if value < 0 {
        if delta > 0 {
            auto_step
        } else {
            -1
        }
    } else if delta > 0 {
        value.saturating_add(step)
    } else {
        let next = value - step;
        if next < 0 {
            -1
        } else {
            next
        }
    }
}

fn step_gpu_layers(value: i64, delta: i32) -> i64 {
    if delta > 0 {
        match value {
            -1 => -2,
            -2 => 0,
            n => (n + 1).min(512),
        }
    } else {
        match value {
            -1 => -1,
            -2 => -1,
            0 => -2,
            n => (n - 1).max(0),
        }
    }
}

fn parse_u32(text: &str, label: &str) -> Result<u32, String> {
    text.parse::<u32>()
        .map_err(|_| format!("{label} must be a whole number."))
}

fn parse_u16(text: &str, label: &str) -> Result<u16, String> {
    text.parse::<u16>()
        .map_err(|_| format!("{label} must be a number between 0 and 65535."))
}

fn parse_f32(text: &str, label: &str) -> Result<f32, String> {
    text.parse::<f32>()
        .map_err(|_| format!("{label} must be a number, for example 0.80."))
}

fn contains(rect: Rect, x: u16, y: u16) -> bool {
    x >= rect.x
        && x < rect.x.saturating_add(rect.width)
        && y >= rect.y
        && y < rect.y.saturating_add(rect.height)
}

fn row_at(rect: Rect, offset: usize, mouse: MouseEvent) -> Option<usize> {
    if !contains(rect, mouse.column, mouse.row) {
        return None;
    }
    Some(offset + (mouse.row - rect.y) as usize)
}

fn row_at_steps(rect: Rect, offset: usize, mouse: MouseEvent, steps: u16) -> Option<usize> {
    if !contains(rect, mouse.column, mouse.row) || steps == 0 {
        return None;
    }
    Some(offset + ((mouse.row - rect.y) / steps) as usize)
}

fn wrap_lines(text: &str) -> Vec<String> {
    text.lines().map(|line| line.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_layer_cycle_is_predictable() {
        assert_eq!(step_gpu_layers(-1, 1), -2);
        assert_eq!(step_gpu_layers(-2, 1), 0);
        assert_eq!(step_gpu_layers(0, 1), 1);
        assert_eq!(step_gpu_layers(1, -1), 0);
        assert_eq!(step_gpu_layers(0, -1), -2);
        assert_eq!(step_gpu_layers(-2, -1), -1);
    }

    #[test]
    fn auto_steps_round_trip() {
        assert_eq!(step_auto(-1, 1, 256, 256), 256);
        assert_eq!(step_auto(256, -1, 256, 256), 0);
        assert_eq!(step_auto(0, -1, 256, 256), -1);
        assert_eq!(step_auto(512, -1, 256, 256), 256);
    }

    #[test]
    fn float_steps_are_rounded() {
        assert!((step_f32(0.8, 1, 0.05, 0.0, 2.0) - 0.85).abs() < 0.001);
        assert!((step_f32(0.0, -1, 0.05, 0.0, 2.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn unknown_row_is_none() {
        let rect = Rect::new(0, 0, 10, 10);
        let mouse = MouseEvent {
            kind: MouseEventKind::Moved,
            column: 50,
            row: 50,
            modifiers: KeyModifiers::NONE,
        };
        assert!(row_at(rect, 0, mouse).is_none());
    }
}
