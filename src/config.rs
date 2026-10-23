use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    #[default]
    LlamaCpp,
    Vllm,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::LlamaCpp => "llama.cpp",
            Backend::Vllm => "vLLM",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Backend::LlamaCpp => Backend::Vllm,
            Backend::Vllm => Backend::LlamaCpp,
        }
    }

    pub fn install_hint(self) -> &'static str {
        match self {
            Backend::LlamaCpp => {
                "Install it with `brew install llama.cpp` (macOS), or set the full \
                 llama-server path in Settings (s)."
            }
            Backend::Vllm => {
                "Install it with `pip install vllm`, or set the full vllm path \
                 (for example a virtualenv path) in Settings (s)."
            }
        }
    }

    pub fn short_install_hint(self) -> &'static str {
        match self {
            Backend::LlamaCpp => "Install: brew install llama.cpp  (or set its path in Settings).",
            Backend::Vllm => "Install: pip install vllm  (or set its path in Settings).",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    #[default]
    Off,
    Low,
    Medium,
    High,
}

impl Effort {
    pub fn label(self) -> &'static str {
        match self {
            Effort::Off => "Off",
            Effort::Low => "Low",
            Effort::Medium => "Medium",
            Effort::High => "High",
        }
    }

    pub fn next(self, delta: i32) -> Self {
        let all = [Effort::Off, Effort::Low, Effort::Medium, Effort::High];
        cycle(&all, self, delta)
    }

    pub fn json_value(self) -> Option<&'static str> {
        match self {
            Effort::Off => None,
            Effort::Low => Some("low"),
            Effort::Medium => Some("medium"),
            Effort::High => Some("high"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlashAttn {
    #[default]
    Auto,
    On,
    Off,
}

impl FlashAttn {
    pub fn label(self) -> &'static str {
        match self {
            FlashAttn::Auto => "Auto",
            FlashAttn::On => "On",
            FlashAttn::Off => "Off",
        }
    }

    pub fn next(self, delta: i32) -> Self {
        cycle(
            &[FlashAttn::Auto, FlashAttn::On, FlashAttn::Off],
            self,
            delta,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpecType {
    #[default]
    Off,
    Draft,
    Mtp,
    Dflash,
    Dspark,
    Ngram,
}

impl SpecType {
    pub fn label(self) -> &'static str {
        match self {
            SpecType::Off => "Off",
            SpecType::Draft => "Draft model",
            SpecType::Mtp => "MTP",
            SpecType::Dflash => "DFlash",
            SpecType::Dspark => "DSpark",
            SpecType::Ngram => "N-gram (no extra model)",
        }
    }

    pub fn next(self, delta: i32, backend: Backend) -> Self {
        if backend == Backend::Vllm {
            return cycle(&[SpecType::Off, SpecType::Draft], self, delta);
        }
        cycle(
            &[
                SpecType::Off,
                SpecType::Draft,
                SpecType::Mtp,
                SpecType::Dflash,
                SpecType::Dspark,
                SpecType::Ngram,
            ],
            self,
            delta,
        )
    }

    pub fn llama_value(self) -> Option<&'static str> {
        match self {
            SpecType::Off => None,
            SpecType::Draft => Some("draft-simple"),
            SpecType::Mtp => Some("draft-mtp"),
            SpecType::Dflash => Some("draft-dflash"),
            SpecType::Dspark => Some("draft-dspark"),
            SpecType::Ngram => Some("ngram-simple"),
        }
    }

    pub fn needs_draft_model(self) -> bool {
        matches!(self, SpecType::Draft)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    #[default]
    Balanced,
    FastLight,
    BestQuality,
    LongDocuments,
    Custom,
}

impl Preset {
    pub const ALL: [Preset; 4] = [
        Preset::Balanced,
        Preset::FastLight,
        Preset::BestQuality,
        Preset::LongDocuments,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Preset::Balanced => "Balanced",
            Preset::FastLight => "Fast & light",
            Preset::BestQuality => "Best quality",
            Preset::LongDocuments => "Long documents",
            Preset::Custom => "Custom",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Preset::Balanced => "A safe starting point for almost any model.",
            Preset::FastLight => "Smaller context and output, snappier replies.",
            Preset::BestQuality => "More context and full GPU offload for better answers.",
            Preset::LongDocuments => "Large context for long files and chat histories.",
            Preset::Custom => "You have changed individual settings.",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub preset: Preset,
    pub context_size: u32,
    pub max_tokens: i64,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub effort: Effort,
    pub reasoning_budget: i64,
    pub gpu_layers: i64,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: u16,
    pub parallel: u16,
    pub flash_attn: FlashAttn,
    pub jinja: bool,
    pub mlock: bool,
    pub seed: i64,
    pub spec_type: SpecType,
    pub draft_model: String,
    pub draft_max: u32,
    pub draft_min: u32,
    pub host: String,
    pub port: u16,
    pub extra_args: String,
    pub vllm_gpu_mem: f32,
    pub vllm_max_seqs: u16,
    pub vllm_batch_tokens: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            preset: Preset::Balanced,
            context_size: 0,
            max_tokens: -1,
            temperature: 0.8,
            top_p: 0.95,
            repeat_penalty: 1.1,
            effort: Effort::Off,
            reasoning_budget: -1,
            gpu_layers: -1,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 0,
            parallel: 1,
            flash_attn: FlashAttn::Auto,
            jinja: true,
            mlock: false,
            seed: -1,
            spec_type: SpecType::Off,
            draft_model: String::new(),
            draft_max: 3,
            draft_min: 0,
            host: "127.0.0.1".to_string(),
            port: 8080,
            extra_args: String::new(),
            vllm_gpu_mem: 0.9,
            vllm_max_seqs: 16,
            vllm_batch_tokens: 8192,
        }
    }
}

impl Settings {
    pub fn apply_preset(&mut self, preset: Preset) {
        match preset {
            Preset::Balanced => {
                self.context_size = 0;
                self.max_tokens = -1;
                self.temperature = 0.8;
                self.top_p = 0.95;
                self.repeat_penalty = 1.1;
                self.gpu_layers = -1;
                self.batch_size = 2048;
                self.ubatch_size = 512;
                self.threads = 0;
                self.parallel = 1;
                self.flash_attn = FlashAttn::Auto;
                self.jinja = true;
                self.mlock = false;
                self.seed = -1;
                self.spec_type = SpecType::Off;
                self.draft_model.clear();
                self.preset = Preset::Balanced;
            }
            Preset::FastLight => {
                self.context_size = 4096;
                self.max_tokens = 1024;
                self.temperature = 0.7;
                self.top_p = 0.9;
                self.repeat_penalty = 1.1;
                self.gpu_layers = -1;
                self.batch_size = 1024;
                self.ubatch_size = 256;
                self.threads = 0;
                self.parallel = 1;
                self.flash_attn = FlashAttn::On;
                self.jinja = true;
                self.mlock = false;
                self.seed = -1;
                self.spec_type = SpecType::Off;
                self.draft_model.clear();
                self.preset = Preset::FastLight;
            }
            Preset::BestQuality => {
                self.context_size = 16384;
                self.max_tokens = -1;
                self.temperature = 0.6;
                self.top_p = 0.9;
                self.repeat_penalty = 1.05;
                self.gpu_layers = -2;
                self.batch_size = 2048;
                self.ubatch_size = 512;
                self.threads = 0;
                self.parallel = 1;
                self.flash_attn = FlashAttn::On;
                self.jinja = true;
                self.mlock = false;
                self.seed = -1;
                self.spec_type = SpecType::Off;
                self.draft_model.clear();
                self.preset = Preset::BestQuality;
            }
            Preset::LongDocuments => {
                self.context_size = 32768;
                self.max_tokens = -1;
                self.temperature = 0.7;
                self.top_p = 0.95;
                self.repeat_penalty = 1.1;
                self.gpu_layers = -2;
                self.batch_size = 2048;
                self.ubatch_size = 512;
                self.threads = 0;
                self.parallel = 1;
                self.flash_attn = FlashAttn::On;
                self.jinja = true;
                self.mlock = false;
                self.seed = -1;
                self.spec_type = SpecType::Off;
                self.draft_model.clear();
                self.preset = Preset::LongDocuments;
            }
            Preset::Custom => {}
        }
    }

    pub fn cycle_preset(&mut self, delta: i32) {
        let current = Preset::ALL
            .iter()
            .position(|p| *p == self.preset)
            .unwrap_or(0) as i32;
        let len = Preset::ALL.len() as i32;
        let next = (current + delta).rem_euclid(len) as usize;
        self.apply_preset(Preset::ALL[next]);
    }

    pub fn mark_custom(&mut self) {
        self.preset = Preset::Custom;
    }

    pub fn sanitize(&mut self) {
        self.context_size = self.context_size.min(1_048_576);
        self.max_tokens = self.max_tokens.max(-1);
        self.temperature = self.temperature.clamp(0.0, 2.0);
        self.top_p = self.top_p.clamp(0.01, 1.0);
        self.repeat_penalty = self.repeat_penalty.clamp(0.5, 2.0);
        self.reasoning_budget = self.reasoning_budget.max(-1);
        self.gpu_layers = self.gpu_layers.clamp(-2, 512);
        self.batch_size = self.batch_size.clamp(32, 65_536);
        self.ubatch_size = self.ubatch_size.clamp(32, 65_536);
        self.threads = self.threads.min(256);
        self.parallel = self.parallel.clamp(1, 64);
        self.draft_max = self.draft_max.min(16);
        self.draft_min = self.draft_min.min(self.draft_max);
        self.host = self.host.trim().to_string();
        if self.host.is_empty() {
            self.host = "127.0.0.1".to_string();
        }
        if self.port == 0 {
            self.port = 8080;
        }
        self.vllm_gpu_mem = self.vllm_gpu_mem.clamp(0.1, 0.99);
        self.vllm_max_seqs = self.vllm_max_seqs.clamp(1, 4096);
        self.vllm_batch_tokens = self.vllm_batch_tokens.clamp(256, 1_048_576);
    }

    pub fn summary(&self, backend: Backend) -> String {
        if backend == Backend::Vllm {
            format!(
                "ctx {}, GPU {:.0}%, {} seqs",
                if self.context_size == 0 {
                    "auto".to_string()
                } else {
                    format_token_count(self.context_size as u64)
                },
                self.vllm_gpu_mem * 100.0,
                self.vllm_max_seqs
            )
        } else {
            let ctx = if self.context_size == 0 {
                "auto".to_string()
            } else {
                format_token_count(self.context_size as u64)
            };
            let gpu = match self.gpu_layers {
                -1 => "auto".to_string(),
                -2 => "all".to_string(),
                n => format!("{n} layers"),
            };
            let spec = if self.spec_type == SpecType::Off {
                String::new()
            } else {
                format!(", {}", self.spec_type.label().to_lowercase())
            };
            format!("ctx {ctx}, GPU {gpu}{spec}")
        }
    }
}

fn cycle<T: Copy + PartialEq>(all: &[T], current: T, delta: i32) -> T {
    let idx = all.iter().position(|v| *v == current).unwrap_or(0) as i32;
    let len = all.len() as i32;
    all[(idx + delta).rem_euclid(len) as usize]
}

pub fn format_token_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1000 {
        format!("{}K", n / 1000)
    } else {
        n.to_string()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub backend: Backend,
    pub models_dir: Option<PathBuf>,
    pub llama_server: String,
    pub vllm: String,
    pub settings: Settings,
    pub per_model: BTreeMap<String, Settings>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::LlamaCpp,
            models_dir: None,
            llama_server: "llama-server".to_string(),
            vllm: "vllm".to_string(),
            settings: Settings::default(),
            per_model: BTreeMap::new(),
        }
    }
}

impl Config {
    pub fn path() -> Option<PathBuf> {
        dirs::config_dir().map(|dir| dir.join("llm").join("config.toml"))
    }

    pub fn log_dir() -> PathBuf {
        Config::path()
            .and_then(|p| p.parent().map(|p| p.join("logs")))
            .unwrap_or_else(|| PathBuf::from("logs"))
    }

    pub fn load() -> (Self, Option<String>) {
        let Some(path) = Config::path() else {
            return (Config::default(), None);
        };
        if !path.exists() {
            return (Config::default(), None);
        }
        match fs::read_to_string(&path) {
            Ok(text) => match toml::from_str::<Config>(&text) {
                Ok(mut cfg) => {
                    cfg.sanitize();
                    (cfg, None)
                }
                Err(err) => (
                    Config::default(),
                    Some(format!("Could not read config; using defaults. ({err})")),
                ),
            },
            Err(err) => (
                Config::default(),
                Some(format!("Could not open config; using defaults. ({err})")),
            ),
        }
    }

    pub fn save(&self) -> Result<()> {
        let Some(path) = Config::path() else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("serializing config")?;
        fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn sanitize(&mut self) {
        self.settings.sanitize();
        for settings in self.per_model.values_mut() {
            settings.sanitize();
        }
        if self.llama_server.trim().is_empty() {
            self.llama_server = "llama-server".to_string();
        }
        if self.vllm.trim().is_empty() {
            self.vllm = "vllm".to_string();
        }
    }

    pub fn binary(&self, backend: Backend) -> &str {
        match backend {
            Backend::LlamaCpp => self.llama_server.trim(),
            Backend::Vllm => self.vllm.trim(),
        }
    }

    pub fn settings_for(&self, path: &Path) -> Settings {
        self.per_model
            .get(&path_key(path))
            .cloned()
            .unwrap_or_else(|| self.settings.clone())
    }

    pub fn store_settings(&mut self, path: &Path, settings: &Settings) {
        self.per_model.insert(path_key(path), settings.clone());
    }
}

pub fn path_key(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

pub fn primary_models_dir(cfg: &Config) -> PathBuf {
    cfg.models_dir.clone().unwrap_or_else(|| {
        dirs::home_dir()
            .map(|home| home.join("models"))
            .unwrap_or_else(|| PathBuf::from("models"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_keep_connection_settings() {
        let mut settings = Settings {
            host: "0.0.0.0".into(),
            port: 1234,
            extra_args: "--foo".into(),
            ..Settings::default()
        };
        settings.apply_preset(Preset::LongDocuments);
        assert_eq!(settings.host, "0.0.0.0");
        assert_eq!(settings.port, 1234);
        assert_eq!(settings.extra_args, "--foo");
        assert_eq!(settings.context_size, 32768);
    }

    #[test]
    fn custom_is_set_after_manual_change() {
        let mut settings = Settings::default();
        settings.mark_custom();
        assert_eq!(settings.preset, Preset::Custom);
    }

    #[test]
    fn sanitize_clamps_values() {
        let mut settings = Settings {
            temperature: 9.0,
            top_p: 0.0,
            port: 0,
            host: "  ".into(),
            ..Settings::default()
        };
        settings.sanitize();
        assert_eq!(settings.temperature, 2.0);
        assert_eq!(settings.top_p, 0.01);
        assert_eq!(settings.port, 8080);
        assert_eq!(settings.host, "127.0.0.1");
    }
}
