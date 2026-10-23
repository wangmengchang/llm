use crate::config::{Backend, Settings, SpecType};
use crate::models::expand_tilde;
use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use std::env;
use std::net::TcpStream;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct LaunchPlan {
    pub program: String,
    pub args: Vec<String>,
}

impl LaunchPlan {
    pub fn command_line(&self) -> String {
        let mut parts = vec![shell_quote(&self.program)];
        parts.extend(self.args.iter().map(|arg| shell_quote(arg)));
        parts.join(" ")
    }
}

pub fn build_plan(
    backend: Backend,
    settings: &Settings,
    model: &Path,
    program: &str,
) -> Result<LaunchPlan> {
    validate(backend, settings, model)?;
    let mut args = match backend {
        Backend::LlamaCpp => llama_args(settings, model)?,
        Backend::Vllm => vllm_args(settings, model)?,
    };
    args.extend(split_args(&settings.extra_args)?);
    let program = expand_tilde(Path::new(program))
        .to_string_lossy()
        .to_string();
    Ok(LaunchPlan { program, args })
}

pub fn validate(backend: Backend, settings: &Settings, model: &Path) -> Result<()> {
    if settings.host.trim().is_empty() {
        bail!("The host address is empty. Set it to 127.0.0.1 in Advanced settings.");
    }
    if settings.port == 0 {
        bail!("The port must be between 1 and 65535.");
    }
    if !model.exists() {
        bail!("The model file no longer exists:\n{}", model.display());
    }
    if settings.spec_type.needs_draft_model() {
        if settings.draft_model.trim().is_empty() {
            bail!(
                "Speculation is set to {}, but no draft model is selected.\n\
                 Pick one under Performance > Draft model, or set Speculation to Off.",
                settings.spec_type.label()
            );
        }
        let draft = expand_tilde(Path::new(settings.draft_model.trim()));
        if !draft.exists() {
            bail!(
                "The draft model file no longer exists:\n{}",
                draft.display()
            );
        }
        if draft == model {
            bail!("The draft model must be a different file from the main model.");
        }
    }
    if backend == Backend::Vllm && !matches!(settings.spec_type, SpecType::Off | SpecType::Draft) {
        bail!(
            "vLLM only supports draft-model speculation here. Choose \"Draft model\" or \"Off\"."
        );
    }
    Ok(())
}

fn round3(value: f32) -> f64 {
    ((value as f64) * 1000.0).round() / 1000.0
}

fn llama_args(settings: &Settings, model: &Path) -> Result<Vec<String>> {
    let mut args = vec![
        "--model".to_string(),
        model.to_string_lossy().to_string(),
        "--host".to_string(),
        settings.host.clone(),
        "--port".to_string(),
        settings.port.to_string(),
    ];

    if settings.context_size > 0 {
        push(&mut args, "-c", settings.context_size);
    }
    if settings.max_tokens >= 0 {
        push(&mut args, "-n", settings.max_tokens);
    }
    push(&mut args, "--temp", format!("{:.2}", settings.temperature));
    push(&mut args, "--top-p", format!("{:.2}", settings.top_p));
    push(
        &mut args,
        "--repeat-penalty",
        format!("{:.2}", settings.repeat_penalty),
    );

    match settings.gpu_layers {
        -1 => {}
        -2 => {
            args.push("-ngl".to_string());
            args.push("all".to_string());
        }
        layers => push(&mut args, "-ngl", layers),
    }
    if settings.threads > 0 {
        push(&mut args, "-t", settings.threads);
    }
    push(&mut args, "-b", settings.batch_size);
    push(&mut args, "-ub", settings.ubatch_size);
    push(&mut args, "-np", settings.parallel);
    match settings.flash_attn {
        crate::config::FlashAttn::Auto => {}
        crate::config::FlashAttn::On => {
            args.push("-fa".to_string());
            args.push("on".to_string());
        }
        crate::config::FlashAttn::Off => {
            args.push("-fa".to_string());
            args.push("off".to_string());
        }
    }
    if !settings.jinja {
        args.push("--no-jinja".to_string());
    }
    if settings.mlock {
        push(&mut args, "-lm", "mlock");
    }
    if settings.seed >= 0 {
        push(&mut args, "-s", settings.seed);
    }
    if let Some(effort) = settings.effort.json_value() {
        let kwargs = json!({ "reasoning_effort": effort });
        push(&mut args, "--chat-template-kwargs", kwargs.to_string());
    }
    if settings.reasoning_budget >= 0 {
        push(&mut args, "--reasoning-budget", settings.reasoning_budget);
    }

    if let Some(spec) = settings.spec_type.llama_value() {
        let has_draft = !settings.draft_model.trim().is_empty();
        if has_draft && settings.spec_type != SpecType::Ngram {
            let draft = expand_tilde(Path::new(settings.draft_model.trim()));
            push(
                &mut args,
                "--model-draft",
                draft.to_string_lossy().to_string(),
            );
        }
        if settings.spec_type != SpecType::Ngram {
            if settings.draft_max > 0 {
                push(&mut args, "--spec-draft-n-max", settings.draft_max);
            }
            if settings.draft_min > 0 {
                push(&mut args, "--spec-draft-n-min", settings.draft_min);
            }
        }
        push(&mut args, "--spec-type", spec);
    }

    Ok(args)
}

fn vllm_args(settings: &Settings, model: &Path) -> Result<Vec<String>> {
    let mut args = vec![
        "serve".to_string(),
        model.to_string_lossy().to_string(),
        "--host".to_string(),
        settings.host.clone(),
        "--port".to_string(),
        settings.port.to_string(),
    ];

    if settings.context_size > 0 {
        push(&mut args, "--max-model-len", settings.context_size);
    }
    push(
        &mut args,
        "--gpu-memory-utilization",
        format!("{:.2}", settings.vllm_gpu_mem),
    );
    push(&mut args, "--max-num-seqs", settings.vllm_max_seqs);
    push(
        &mut args,
        "--max-num-batched-tokens",
        settings.vllm_batch_tokens,
    );
    if settings.seed >= 0 {
        push(&mut args, "--seed", settings.seed);
    }

    let mut generation = Map::new();
    generation.insert("temperature".into(), json!(round3(settings.temperature)));
    generation.insert("top_p".into(), json!(round3(settings.top_p)));
    generation.insert(
        "repetition_penalty".into(),
        json!(round3(settings.repeat_penalty)),
    );
    if settings.max_tokens >= 0 {
        generation.insert("max_new_tokens".into(), json!(settings.max_tokens));
    }
    push(
        &mut args,
        "--override-generation-config",
        Value::Object(generation).to_string(),
    );

    if let Some(effort) = settings.effort.json_value() {
        let kwargs = json!({ "reasoning_effort": effort });
        push(
            &mut args,
            "--default-chat-template-kwargs",
            kwargs.to_string(),
        );
    }

    if settings.spec_type == SpecType::Draft {
        let draft = expand_tilde(Path::new(settings.draft_model.trim()));
        push(
            &mut args,
            "--speculative-model",
            draft.to_string_lossy().to_string(),
        );
        if settings.draft_max > 0 {
            push(&mut args, "--num-speculative-tokens", settings.draft_max);
        }
    }

    Ok(args)
}

fn push<T: ToString>(args: &mut Vec<String>, flag: &str, value: T) {
    args.push(flag.to_string());
    args.push(value.to_string());
}

pub fn split_args(input: &str) -> Result<Vec<String>> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut has_content = false;

    for ch in input.chars() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                } else {
                    current.push(ch);
                    has_content = true;
                }
            }
            None => {
                if ch == '\'' || ch == '"' {
                    quote = Some(ch);
                    has_content = true;
                } else if ch.is_whitespace() {
                    if has_content {
                        args.push(std::mem::take(&mut current));
                        has_content = false;
                    }
                } else {
                    current.push(ch);
                    has_content = true;
                }
            }
        }
    }
    if quote.is_some() {
        bail!("Extra arguments contain an unmatched quote.");
    }
    if has_content {
        args.push(current);
    }
    Ok(args)
}

pub fn binary_available(name: &str) -> bool {
    let name = name.trim();
    if name.is_empty() {
        return false;
    }
    let path = expand_tilde(Path::new(name));
    if name.contains('/') || name.contains('\\') {
        return path.is_file();
    }
    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    for dir in env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return true;
        }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{name}.exe"));
            if exe.is_file() {
                return true;
            }
        }
    }
    false
}

pub fn port_open(port: u16) -> bool {
    TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(120),
    )
    .is_ok()
}

pub fn find_free_port(start: u16) -> u16 {
    for offset in 0..24u16 {
        let port = start.saturating_add(offset);
        if port > 0 && !port_open(port) {
            return port;
        }
    }
    start
}

pub fn open_path(path: &Path) -> std::io::Result<()> {
    open_target(&path.to_string_lossy())
}

pub fn open_url(url: &str) -> std::io::Result<()> {
    open_target(url)
}

fn open_target(target: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut cmd = Command::new("open");
        cmd.arg(target);
        cmd
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", "start", "", target]);
        cmd
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut cmd = Command::new("xdg-open");
        cmd.arg(target);
        cmd
    };
    command.spawn().map(|_| ())
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-+=:,{}[]@".contains(c))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Effort, Preset};

    fn settings() -> Settings {
        Settings::default()
    }

    fn temp_model(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("llm-test-{}-{name}", std::process::id()));
        std::fs::write(&path, b"test").unwrap();
        path
    }

    #[test]
    fn builds_safe_llama_command() {
        let mut s = settings();
        s.port = 8080;
        s.context_size = 8192;
        s.max_tokens = 512;
        s.gpu_layers = -2;
        s.flash_attn = crate::config::FlashAttn::On;
        s.effort = Effort::Medium;
        s.mark_custom();

        let model = temp_model("safe.gguf");
        let plan = build_plan(Backend::LlamaCpp, &s, &model, "llama-server").unwrap();

        assert_eq!(plan.args[0], "--model");
        assert!(plan.args.contains(&"-c".to_string()));
        assert!(plan.args.contains(&"8192".to_string()));
        assert!(plan.args.contains(&"-ngl".to_string()));
        assert!(plan.args.contains(&"all".to_string()));
        assert!(plan.command_line().contains("--chat-template-kwargs"));
        let _ = std::fs::remove_file(model);
    }

    #[test]
    fn refuses_draft_without_draft_model() {
        let mut s = settings();
        s.spec_type = SpecType::Draft;
        let model = temp_model("draft.gguf");
        let err = build_plan(Backend::LlamaCpp, &s, &model, "llama-server").unwrap_err();
        assert!(err.to_string().contains("draft model"));
        let _ = std::fs::remove_file(model);
    }

    #[test]
    fn builds_vllm_command() {
        let mut s = settings();
        s.apply_preset(Preset::FastLight);
        s.port = 9000;
        let model = std::env::temp_dir().join(format!("llm-test-{}-dir", std::process::id()));
        std::fs::create_dir_all(&model).unwrap();
        let plan = build_plan(Backend::Vllm, &s, &model, "vllm").unwrap();
        assert_eq!(plan.args[0], "serve");
        assert!(plan.args.contains(&"--max-model-len".to_string()));
        assert!(plan
            .args
            .contains(&"--override-generation-config".to_string()));
        let _ = std::fs::remove_dir_all(model);
    }

    #[test]
    fn splits_extra_arguments_with_quotes() {
        let args = split_args("--foo bar --baz 'hello world' \"x y\"").unwrap();
        assert_eq!(args, vec!["--foo", "bar", "--baz", "hello world", "x y"]);
        assert!(split_args("--foo 'oops").is_err());
    }

    #[test]
    fn quotes_command_line_parts() {
        assert_eq!(shell_quote("/tmp/a b.gguf"), "'/tmp/a b.gguf'");
        assert_eq!(shell_quote("--model"), "--model");
    }
}
