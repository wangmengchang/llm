use crate::backend::{port_open, LaunchPlan};
use crate::config::Backend;
use anyhow::{Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerStatus {
    Idle,
    Starting,
    Ready,
    Exited(String),
}

#[derive(Clone, Debug)]
pub struct RunInfo {
    pub model_name: String,
    pub model_path: PathBuf,
    pub backend: Backend,
    pub host: String,
    pub port: u16,
    pub context: String,
}

#[derive(Debug)]
pub struct ServerManager {
    child: Option<Child>,
    status: ServerStatus,
    info: Option<RunInfo>,
    log_path: Option<PathBuf>,
    logs: Vec<String>,
    log_len: u64,
    started: Option<Instant>,
    last_health_check: Instant,
}

impl Default for ServerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerManager {
    pub fn new() -> Self {
        Self {
            child: None,
            status: ServerStatus::Idle,
            info: None,
            log_path: None,
            logs: Vec::new(),
            log_len: 0,
            started: None,
            last_health_check: Instant::now(),
        }
    }

    pub fn start(&mut self, plan: &LaunchPlan, info: RunInfo, log_dir: &Path) -> Result<()> {
        if self.running() {
            anyhow::bail!("A server is already running.");
        }
        fs::create_dir_all(log_dir)
            .with_context(|| format!("creating log folder {}", log_dir.display()))?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let log_path = log_dir.join(format!("server-{stamp}.log"));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&log_path)
            .with_context(|| format!("creating log file {}", log_path.display()))?;
        let command_line = plan.command_line();
        writeln!(file, "# llm started this server")?;
        writeln!(file, "# {command_line}")?;
        writeln!(file)?;
        file.flush()?;

        let stdout = file.try_clone().context("preparing the server log")?;
        let stderr = file.try_clone().context("preparing the server log")?;

        let mut command = Command::new(&plan.program);
        command
            .args(&plan.args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let child = command.spawn().with_context(|| {
            format!(
                "could not start `{}`. Is it installed and on your PATH?",
                plan.program
            )
        })?;

        self.child = Some(child);
        self.status = ServerStatus::Starting;
        self.info = Some(info);
        self.log_path = Some(log_path);
        self.logs = vec![format!("$ {command_line}")];
        self.log_len = 0;
        self.started = Some(Instant::now());
        self.last_health_check = Instant::now();
        self.read_logs();
        Ok(())
    }

    pub fn poll(&mut self) -> Option<String> {
        let mut exit_message = None;

        if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    self.child = None;
                    let label = if status.success() {
                        "The server stopped.".to_string()
                    } else {
                        match status.code() {
                            Some(code) => format!("The server exited with code {code}."),
                            None => "The server was stopped.".to_string(),
                        }
                    };
                    self.status = ServerStatus::Exited(label.clone());
                    exit_message = Some(label);
                }
                Ok(None) => {
                    if self.status == ServerStatus::Starting
                        && self.last_health_check.elapsed() >= Duration::from_millis(300)
                    {
                        self.last_health_check = Instant::now();
                        if let Some(info) = &self.info {
                            if port_open(info.port) {
                                self.status = ServerStatus::Ready;
                            }
                        }
                    }
                }
                Err(_) => {
                    self.child = None;
                    self.status = ServerStatus::Exited("Could not check the server.".into());
                }
            }
        }

        self.read_logs();
        exit_message
    }

    fn read_logs(&mut self) {
        let Some(path) = &self.log_path else {
            return;
        };
        let Ok(meta) = fs::metadata(path) else {
            return;
        };
        if meta.len() == self.log_len {
            return;
        }
        let Ok(mut file) = File::open(path) else {
            return;
        };
        let start = meta.len().saturating_sub(32 * 1024);
        if file.seek(SeekFrom::Start(start)).is_err() {
            return;
        }
        let mut buf = Vec::new();
        if file.read_to_end(&mut buf).is_err() {
            return;
        }
        let text = String::from_utf8_lossy(&buf);
        let lines: Vec<String> = text
            .lines()
            .map(|line| clean_log_line(&line.replace('\t', "    ")))
            .collect();
        self.logs = lines;
        self.log_len = meta.len();
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            #[cfg(unix)]
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
            self.status = ServerStatus::Exited("The server was stopped by you.".into());
        }
    }

    pub fn clear(&mut self) {
        if !self.running() {
            self.status = ServerStatus::Idle;
            self.info = None;
            self.log_path = None;
            self.logs.clear();
            self.log_len = 0;
            self.started = None;
        }
    }

    pub fn running(&self) -> bool {
        self.child.is_some()
    }

    pub fn status(&self) -> &ServerStatus {
        &self.status
    }

    pub fn info(&self) -> Option<&RunInfo> {
        self.info.as_ref()
    }

    pub fn logs(&self) -> &[String] {
        &self.logs
    }

    pub fn log_path(&self) -> Option<&Path> {
        self.log_path.as_deref()
    }

    pub fn uptime(&self) -> Option<Duration> {
        self.started.map(|started| started.elapsed())
    }

    pub fn endpoint(&self) -> Option<String> {
        let info = self.info.as_ref()?;
        let host = match info.host.as_str() {
            "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
            other => other,
        };
        Some(format!("http://{host}:{}", info.port))
    }

    pub fn api_endpoint(&self) -> Option<String> {
        self.endpoint().map(|endpoint| format!("{endpoint}/v1"))
    }
}

impl Drop for ServerManager {
    fn drop(&mut self) {
        if self.running() {
            self.stop();
        }
    }
}

fn clean_log_line(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\r' {
            continue;
        }
        if ch == '\u{1b}' {
            match chars.peek() {
                Some('[') => {
                    chars.next();
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    chars.next();
                    let mut escaped = false;
                    for next in chars.by_ref() {
                        if next == '\u{7}' {
                            break;
                        }
                        if escaped {
                            if next == '\\' {
                                break;
                            }
                            escaped = false;
                        }
                        if next == '\u{1b}' {
                            escaped = true;
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        if ch.is_control() {
            continue;
        }
        out.push(ch);
    }
    out.trim_end().to_string()
}

pub fn format_uptime(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uptime_formats() {
        assert_eq!(format_uptime(Duration::from_secs(42)), "42s");
        assert_eq!(format_uptime(Duration::from_secs(125)), "2m 05s");
        assert_eq!(format_uptime(Duration::from_secs(3725)), "1h 02m");
    }

    #[test]
    fn strips_ansi_and_carriage_returns() {
        assert_eq!(clean_log_line("\u{1b}[32mready\u{1b}[0m"), "ready");
        assert_eq!(clean_log_line("loading 50%\r"), "loading 50%");
        assert_eq!(clean_log_line("plain text"), "plain text");
    }
}
