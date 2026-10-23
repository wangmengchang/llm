mod app;
mod backend;
mod config;
mod fields;
mod gguf;
mod models;
mod server;
mod ui;

use anyhow::{bail, Result};
use app::{App, Tone};
use backend::binary_available;
use clap::Parser;
use config::{Backend, Config};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use models::expand_tilde;
use ratatui::backend::{Backend as BackendTrait, CrosstermBackend};
use ratatui::Terminal;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(
    name = "llm",
    version,
    about = "Local LLM Manager - run local models from a friendly terminal UI"
)]
struct Cli {
    #[arg(long, value_name = "DIR", help = "Add a folder to search for models")]
    models_dir: Option<PathBuf>,

    #[arg(
        long,
        value_name = "BACKEND",
        help = "Backend to use: llama.cpp or vllm"
    )]
    backend: Option<String>,

    #[arg(long, help = "List found models and exit")]
    list_models: bool,

    #[arg(long, help = "Do not capture the mouse")]
    no_mouse: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (mut cfg, warning) = Config::load();

    if let Some(dir) = &cli.models_dir {
        cfg.models_dir = Some(expand_tilde(dir));
    }
    if let Some(backend) = &cli.backend {
        cfg.backend = parse_backend(backend)?;
    }

    if cli.list_models {
        return list_models(&cfg);
    }

    run_tui(cfg, warning, !cli.no_mouse)
}

fn parse_backend(value: &str) -> Result<Backend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "llama.cpp" | "llamacpp" | "llama" | "llama-cpp" => Ok(Backend::LlamaCpp),
        "vllm" | "v-llm" => Ok(Backend::Vllm),
        other => bail!("Unknown backend `{other}`. Use `llama.cpp` or `vllm`."),
    }
}

fn list_models(cfg: &Config) -> Result<()> {
    let result = models::scan(cfg.backend, cfg);
    println!(
        "Found {} for {}.\n",
        models::count_label(result.models.len()),
        cfg.backend.label()
    );
    if !result.models.is_empty() {
        println!(
            "{:<36} {:<11} {:>9}  {:<30} PATH",
            "MODEL", "SOURCE", "SIZE", "DETAILS"
        );
        for model in &result.models {
            let mut details = model.tags();
            if let Some(arch) = &model.arch {
                details.push(arch.clone());
            }
            println!(
                "{:<36} {:<11} {:>9}  {:<30} {}",
                model.name,
                format!("[{}]", model.source.label()),
                models::human_size(model.size_bytes),
                details.join(" "),
                model.path.display()
            );
        }
    } else {
        println!("Searched these folders:");
        for path in &result.searched {
            println!("  {}", path.display());
        }
        println!("\nPut a .gguf model in one of them, or run `llm` and press s to set a folder.");
        let program = cfg.binary(cfg.backend);
        if !binary_available(program) {
            println!(
                "\n{program} is not installed. {}",
                cfg.backend.install_hint()
            );
        }
    }
    Ok(())
}

fn run_tui(cfg: Config, warning: Option<String>, mouse: bool) -> Result<()> {
    install_panic_hook();
    enable_raw_mode()?;
    let _guard = TerminalGuard { mouse };

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    if mouse {
        execute!(stdout, EnableMouseCapture)?;
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = App::new(cfg);
    if let Some(warning) = warning {
        app.notify(warning, Tone::Warn);
    }
    let program = app.cfg.binary(app.cfg.backend).to_string();
    if !binary_available(&program) {
        app.notify(
            format!("`{program}` was not found. Press s to set a path, or install it."),
            Tone::Warn,
        );
    }

    let result = event_loop(&mut terminal, &mut app);
    let shutdown = app.shutdown();
    restore_terminal(&mut terminal, mouse);
    drop(_guard);

    result?;
    if let Err(err) = shutdown {
        eprintln!("llm: could not save settings: {err}");
    }
    Ok(())
}

fn event_loop<B: BackendTrait>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|frame| ui::draw(frame, app))?;
        if event::poll(Duration::from_millis(120))? {
            match event::read()? {
                Event::Key(key) => app.on_key(key),
                Event::Mouse(mouse) => app.on_mouse(mouse),
                _ => {}
            }
        }
        app.on_tick();
        if app.quit {
            break;
        }
    }
    Ok(())
}

fn restore_terminal<B: BackendTrait + std::io::Write>(terminal: &mut Terminal<B>, mouse: bool) {
    let _ = disable_raw_mode();
    if mouse {
        let _ = execute!(terminal.backend_mut(), DisableMouseCapture);
    }
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
}

fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            crossterm::cursor::Show
        );
        original(info);
    }));
}

struct TerminalGuard {
    mouse: bool,
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        if self.mouse {
            let _ = execute!(stdout, DisableMouseCapture);
        }
        let _ = execute!(stdout, LeaveAlternateScreen, crossterm::cursor::Show);
    }
}
