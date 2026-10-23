use crate::config::{Backend, Config};
use crate::gguf;
use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSource {
    Local,
    Ollama,
    LmStudio,
}

impl ModelSource {
    pub fn label(self) -> &'static str {
        match self {
            ModelSource::Local => "local",
            ModelSource::Ollama => "Ollama",
            ModelSource::LmStudio => "LM Studio",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Gguf,
    HfDir,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub source: ModelSource,
    pub kind: ModelKind,
    pub quant: Option<String>,
    pub params: Option<String>,
    pub arch: Option<String>,
    pub native_ctx: Option<u64>,
    pub meta_name: Option<String>,
}

impl Model {
    pub fn context_label(&self) -> Option<String> {
        self.native_ctx
            .map(crate::config::format_token_count)
            .filter(|s| !s.is_empty())
    }

    pub fn tags(&self) -> Vec<String> {
        let mut tags = Vec::new();
        if let Some(params) = &self.params {
            tags.push(params.clone());
        }
        if let Some(quant) = &self.quant {
            tags.push(quant.clone());
        }
        if let Some(ctx) = &self.native_ctx {
            tags.push(format!("{} ctx", crate::config::format_token_count(*ctx)));
        }
        tags
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScanResult {
    pub models: Vec<Model>,
    pub searched: Vec<PathBuf>,
    pub notes: Vec<String>,
}

pub fn scan(backend: Backend, cfg: &Config) -> ScanResult {
    let mut result = ScanResult::default();
    let mut seen = HashSet::new();
    let mut roots: Vec<(PathBuf, ModelSource)> = Vec::new();

    if let Some(dir) = &cfg.models_dir {
        roots.push((expand_tilde(dir), ModelSource::Local));
    }
    if let Ok(dir) = env::var("LLM_MODELS_DIR") {
        if !dir.trim().is_empty() {
            roots.push((expand_tilde(Path::new(dir.trim())), ModelSource::Local));
        }
    }
    if let Ok(cwd) = env::current_dir() {
        roots.push((cwd.join("models"), ModelSource::Local));
    }
    if let Some(home) = dirs::home_dir() {
        roots.push((home.join("models"), ModelSource::Local));
        if backend == Backend::LlamaCpp {
            roots.push((home.join(".cache").join("llama.cpp"), ModelSource::Local));
        }
        roots.push((home.join(".lmstudio").join("models"), ModelSource::LmStudio));
        roots.push((
            home.join(".cache").join("lm-studio").join("models"),
            ModelSource::LmStudio,
        ));
        roots.push((
            home.join("Library")
                .join("Application Support")
                .join("LM Studio")
                .join("models"),
            ModelSource::LmStudio,
        ));
    }

    let mut seen_roots = HashSet::new();
    for (root, source) in roots {
        if !seen_roots.insert(normalize(&root)) {
            continue;
        }
        if !result.searched.contains(&root) {
            result.searched.push(root.clone());
        }
        if !root.exists() {
            continue;
        }
        match backend {
            Backend::LlamaCpp => collect_gguf(
                &root,
                source,
                &mut result.models,
                &mut seen,
                0,
                &mut result.notes,
            ),
            Backend::Vllm => collect_hf_dirs(&root, source, &mut result.models, &mut seen, 0),
        }
    }

    if backend == Backend::LlamaCpp {
        for (root, _source) in ollama_roots() {
            if !seen_roots.insert(normalize(&root)) {
                continue;
            }
            if !result.searched.contains(&root) {
                result.searched.push(root.clone());
            }
            if root.exists() {
                collect_ollama(&root, &mut result.models, &mut seen, &mut result.notes);
            }
        }
    }

    result.models.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.path.cmp(&b.path))
    });
    result.models.dedup_by(|a, b| a.path == b.path);
    result
}

fn collect_gguf(
    dir: &Path,
    source: ModelSource,
    out: &mut Vec<Model>,
    seen: &mut HashSet<PathBuf>,
    depth: usize,
    notes: &mut Vec<String>,
) {
    if depth > 8 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_dir() {
            if should_skip_dir(&path) {
                continue;
            }
            collect_gguf(&path, source, out, seen, depth + 1, notes);
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if !is_gguf {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        if is_projector(&stem) || is_later_shard(&stem) {
            continue;
        }
        let canonical = normalize(&path);
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let Some(info) = gguf::read_info(&path) else {
            notes.push(format!("Skipped unreadable GGUF: {}", path.display()));
            continue;
        };
        out.push(Model {
            name: stem.clone(),
            path: canonical,
            size_bytes: meta.len(),
            source,
            kind: ModelKind::Gguf,
            quant: quant_from_name(&stem),
            params: params_from_name(&stem),
            arch: info.arch,
            native_ctx: info.context,
            meta_name: info.name.filter(|name| !name.is_empty() && *name != stem),
        });
    }
}

fn collect_hf_dirs(
    dir: &Path,
    source: ModelSource,
    out: &mut Vec<Model>,
    seen: &mut HashSet<PathBuf>,
    depth: usize,
) {
    if depth > 8 {
        return;
    }
    if dir.join("config.json").is_file() {
        let canonical = normalize(dir);
        if !seen.insert(canonical.clone()) {
            return;
        }
        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        let size = directory_size(dir, 0);
        out.push(Model {
            name,
            path: canonical,
            size_bytes: size,
            source,
            kind: ModelKind::HfDir,
            quant: None,
            params: None,
            arch: None,
            native_ctx: None,
            meta_name: None,
        });
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && !should_skip_dir(&path) {
            collect_hf_dirs(&path, source, out, seen, depth + 1);
        }
    }
}

fn collect_ollama(
    root: &Path,
    out: &mut Vec<Model>,
    seen: &mut HashSet<PathBuf>,
    _notes: &mut Vec<String>,
) {
    let manifests = root.join("manifests");
    let blobs = root.join("blobs");
    if !manifests.is_dir() || !blobs.is_dir() {
        return;
    }
    let mut files = Vec::new();
    collect_manifest_files(&manifests, &mut files, 0);
    for manifest in files {
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Some(layers) = json.get("layers").and_then(|v| v.as_array()) else {
            continue;
        };
        let Some(layer) = layers.iter().find(|layer| {
            layer
                .get("mediaType")
                .and_then(|v| v.as_str())
                .map(|m| m.ends_with("image.model"))
                .unwrap_or(false)
        }) else {
            continue;
        };
        let Some(digest) = layer.get("digest").and_then(|v| v.as_str()) else {
            continue;
        };
        let blob_name = digest.replace(':', "-");
        let blob = blobs.join(blob_name);
        if !blob.is_file() {
            continue;
        }
        let canonical = normalize(&blob);
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let name = ollama_name(&manifests, &manifest);
        let info = gguf::read_info(&blob).unwrap_or_default();
        let size = fs::metadata(&blob).map(|m| m.len()).unwrap_or(0);
        let params = params_from_name(&name);
        out.push(Model {
            name,
            path: canonical,
            size_bytes: size,
            source: ModelSource::Ollama,
            kind: ModelKind::Gguf,
            quant: None,
            params,
            arch: info.arch,
            native_ctx: info.context,
            meta_name: info.name,
        });
    }
}

fn collect_manifest_files(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 6 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_manifest_files(&path, out, depth + 1);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn ollama_name(manifests: &Path, manifest: &Path) -> String {
    let Ok(rel) = manifest.strip_prefix(manifests) else {
        return "ollama model".to_string();
    };
    let parts: Vec<String> = rel
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(v) => Some(v.to_string_lossy().to_string()),
            _ => None,
        })
        .collect();
    match parts.len() {
        0 => "ollama model".to_string(),
        1 => parts[0].clone(),
        _ => format!("{}:{}", parts[parts.len() - 2], parts[parts.len() - 1]),
    }
}

fn ollama_roots() -> Vec<(PathBuf, ModelSource)> {
    let mut roots = Vec::new();
    if let Ok(dir) = env::var("OLLAMA_MODELS") {
        if !dir.trim().is_empty() {
            roots.push((expand_tilde(Path::new(dir.trim())), ModelSource::Ollama));
        }
    }
    if let Some(home) = dirs::home_dir() {
        roots.push((home.join(".ollama").join("models"), ModelSource::Ollama));
    }
    roots
}

fn is_projector(stem: &str) -> bool {
    stem.to_ascii_lowercase().contains("mmproj")
}

fn is_later_shard(stem: &str) -> bool {
    let lower = stem.to_ascii_lowercase();
    let Some(position) = lower.rfind("-of-") else {
        return false;
    };
    let prefix = &lower[..position];
    let Some(shard) = prefix.rsplit('-').next() else {
        return false;
    };
    if shard.is_empty() || !shard.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    shard
        .parse::<u32>()
        .map(|number| number > 1)
        .unwrap_or(false)
}

fn should_skip_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    matches!(
        name,
        ".git" | "node_modules" | ".venv" | "venv" | "__pycache__" | ".Trash"
    )
}

fn directory_size(dir: &Path, depth: usize) -> u64 {
    if depth > 6 {
        return 0;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_file() {
            total = total.saturating_add(meta.len());
        } else if meta.is_dir() {
            total = total.saturating_add(directory_size(&entry.path(), depth + 1));
        }
    }
    total
}

fn normalize(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

pub fn expand_tilde(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text == "~" {
        return dirs::home_dir().unwrap_or_else(|| path.to_path_buf());
    }
    if let Some(rest) = text.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    path.to_path_buf()
}

pub fn count_label(count: usize) -> String {
    if count == 1 {
        "1 model".to_string()
    } else {
        format!("{count} models")
    }
}

pub fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn quant_from_name(name: &str) -> Option<String> {
    const QUANTS: [&str; 33] = [
        "IQ2_XXS", "IQ2_XS", "IQ1_S", "IQ1_M", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S",
        "IQ3_M", "IQ4_NL", "IQ4_XS", "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S",
        "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q6_K", "Q8_0", "MXFP4",
        "BF16", "F16", "F32", "Q4_K", "Q5_K",
    ];
    let upper = name.to_ascii_uppercase();
    QUANTS
        .iter()
        .find(|quant| upper.contains(**quant))
        .map(|quant| quant.to_string())
}

pub fn params_from_name(name: &str) -> Option<String> {
    let chars: Vec<char> = name.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'x' || chars[i] == 'X') {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < chars.len()
                && (chars[i] == 'b' || chars[i] == 'B' || chars[i] == 'm' || chars[i] == 'M')
            {
                let next_alnum = chars
                    .get(i + 1)
                    .map(|c| c.is_ascii_alphanumeric())
                    .unwrap_or(false);
                if !next_alnum {
                    let mut label: String = chars[start..=i].iter().collect();
                    label = label.replace('b', "B").replace('m', "M");
                    return Some(label);
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_quantization() {
        assert_eq!(
            quant_from_name("Qwen3-8B-Q4_K_M").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            quant_from_name("Llama-3.2-3B-Instruct-Q8_0").as_deref(),
            Some("Q8_0")
        );
        assert_eq!(quant_from_name("model-bf16").as_deref(), Some("BF16"));
    }

    #[test]
    fn extracts_parameter_counts() {
        assert_eq!(params_from_name("Qwen3-8B-Q4_K_M").as_deref(), Some("8B"));
        assert_eq!(
            params_from_name("llama-3.2-3b-instruct").as_deref(),
            Some("3B")
        );
        assert_eq!(params_from_name("Mixtral-8x7B").as_deref(), Some("8x7B"));
        assert_eq!(params_from_name("Q4_K_M").as_deref(), None);
    }

    #[test]
    fn skips_projectors_and_later_shards() {
        assert!(is_projector("mmproj-model-f16"));
        assert!(!is_projector("Qwen3-8B-Q4_K_M"));
        assert!(is_later_shard("model-00002-of-00003"));
        assert!(!is_later_shard("model-00001-of-00003"));
        assert!(!is_later_shard("Qwen3-8B-Q4_K_M"));
    }

    #[test]
    fn formats_sizes() {
        assert_eq!(human_size(999), "999 B");
        assert_eq!(human_size(1500), "1.5 KB");
        assert_eq!(human_size(4_700_000_000), "4.7 GB");
    }

    #[test]
    fn ollama_name_uses_last_two_components() {
        let root = Path::new("/m/manifests");
        let manifest = Path::new("/m/manifests/registry.ollama.ai/library/llama3.2/3b");
        assert_eq!(ollama_name(root, manifest), "llama3.2:3b");
    }
}
