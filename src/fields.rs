use crate::config::Backend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Preset,
    Context,
    MaxTokens,
    Temperature,
    TopP,
    RepeatPenalty,
    Effort,
    ReasoningBudget,
    GpuLayers,
    Batch,
    Ubatch,
    Threads,
    Parallel,
    FlashAttn,
    Jinja,
    Mlock,
    Seed,
    SpecType,
    DraftModel,
    DraftMax,
    DraftMin,
    Host,
    Port,
    ExtraArgs,
    VllmGpuMem,
    VllmMaxSeqs,
    VllmBatchTokens,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Text,
    Integer,
    Float,
    Bool,
    Cycle,
    Preset,
    GpuLayers,
}

impl Field {
    pub fn label(self) -> &'static str {
        match self {
            Field::Preset => "Preset",
            Field::Context => "Context length",
            Field::MaxTokens => "Max reply tokens",
            Field::Temperature => "Temperature",
            Field::TopP => "Top-p",
            Field::RepeatPenalty => "Repeat penalty",
            Field::Effort => "Reasoning effort",
            Field::ReasoningBudget => "Reasoning budget",
            Field::GpuLayers => "GPU layers",
            Field::Batch => "Batch size",
            Field::Ubatch => "Micro-batch",
            Field::Threads => "CPU threads",
            Field::Parallel => "Parallel chats",
            Field::FlashAttn => "Flash attention",
            Field::Jinja => "Chat templates",
            Field::Mlock => "Keep model in RAM",
            Field::Seed => "Random seed",
            Field::SpecType => "Speculation",
            Field::DraftModel => "Draft model",
            Field::DraftMax => "Draft tokens",
            Field::DraftMin => "Min draft tokens",
            Field::Host => "Host",
            Field::Port => "Port",
            Field::ExtraArgs => "Extra arguments",
            Field::VllmGpuMem => "GPU memory",
            Field::VllmMaxSeqs => "Max concurrent requests",
            Field::VllmBatchTokens => "Batched tokens",
        }
    }

    pub fn kind(self) -> FieldKind {
        match self {
            Field::Preset => FieldKind::Preset,
            Field::Effort | Field::FlashAttn | Field::SpecType | Field::DraftModel => {
                FieldKind::Cycle
            }
            Field::Jinja | Field::Mlock => FieldKind::Bool,
            Field::GpuLayers => FieldKind::GpuLayers,
            Field::Temperature | Field::TopP | Field::RepeatPenalty | Field::VllmGpuMem => {
                FieldKind::Float
            }
            Field::Host | Field::ExtraArgs => FieldKind::Text,
            _ => FieldKind::Integer,
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            Field::Preset => {
                "Ready-made combinations of settings. Left/Right cycles through them. \
                 Start with Balanced and only change things if you need to."
            }
            Field::Context => {
                "How much text, in tokens, the model can remember in one conversation. \
                 Auto uses the model's built-in limit. Bigger contexts need more RAM or VRAM. \
                 If loading fails, lower this first."
            }
            Field::MaxTokens => {
                "The most tokens the model may write in one reply. Auto means no extra cap. \
                 Lower values stop runaway answers and save time."
            }
            Field::Temperature => {
                "Creativity. Lower is more focused and repeatable; higher is more varied. \
                 Coding usually works best around 0.2-0.6, chat around 0.7-0.9."
            }
            Field::TopP => {
                "Limits word choices to the most likely options. 0.95 is a good default. \
                 Leave it alone unless you know why you are changing it."
            }
            Field::RepeatPenalty => {
                "Discourages repeating the same words. 1.0 disables it. Too high makes text \
                 unnatural or terse."
            }
            Field::Effort => {
                "Tells reasoning models how hard to think. It is passed to the model's chat \
                 template, so only models that support reasoning effort will honor it. \
                 Off leaves the model's own default alone."
            }
            Field::ReasoningBudget => {
                "Caps how many tokens a reasoning model may spend thinking. -1 is unlimited, \
                 0 disables thinking. A lower budget is faster but less deliberate."
            }
            Field::GpuLayers => {
                "How much of the model to place on your graphics card. Auto lets llama.cpp \
                 decide; All is fastest but needs enough VRAM. 0 runs on the CPU. If you see \
                 out-of-memory errors, set this back to Auto."
            }
            Field::Batch => {
                "How much prompt text is processed at once. Larger can be faster but uses more \
                 memory. 2048 is a good default."
            }
            Field::Ubatch => {
                "Smaller internal batch used on the GPU. 512 is a good default. Lower it if you \
                 run out of memory."
            }
            Field::Threads => "CPU threads used for generation. Auto is best on most machines.",
            Field::Parallel => {
                "How many conversations the server can handle at once. 1 is right for personal \
                 use. Each extra slot also needs more context memory."
            }
            Field::FlashAttn => {
                "A faster attention implementation. Auto lets the backend decide. On can speed \
                 up long contexts; use Off if your backend crashes with it."
            }
            Field::Jinja => {
                "Uses the chat template embedded in the model, so roles, tool calls, and \
                 thinking work correctly. Keep this on unless you have a reason not to."
            }
            Field::Mlock => {
                "Locks the model in RAM so it is never swapped to disk. Only enable this if you \
                 have plenty of RAM."
            }
            Field::Seed => {
                "Makes random sampling repeatable. Auto is random every time. Set a number if you \
                 want the same output from the same prompt."
            }
            Field::SpecType => {
                "Speculative decoding speeds up generation by guessing ahead. Off is safest. \
                 Draft model uses a second small model; MTP, DFlash, and DSpark need a compatible \
                 model; N-gram needs no extra model and helps with repetitive text or code. If \
                 answers get worse, turn this off."
            }
            Field::DraftModel => {
                "A smaller model that guesses tokens for speculative decoding. Pick one from the \
                 same model family when possible. Left/Right cycles through models llm found."
            }
            Field::DraftMax => {
                "How many tokens the draft model may guess ahead. 3 is a good default. Higher \
                 values can help or hurt depending on the models."
            }
            Field::DraftMin => {
                "Minimum draft length before speculation is used. 0 leaves it to the backend."
            }
            Field::Host => {
                "The network address the server listens on. 127.0.0.1 keeps it private to this \
                 computer. Only use 0.0.0.0 if you intentionally want to share it on your network."
            }
            Field::Port => {
                "The local port for the web server. 8080 is conventional. If it is busy, llm \
                 automatically picks a nearby free port when launching."
            }
            Field::ExtraArgs => {
                "Extra command-line flags for advanced users, separated by spaces. Leave this \
                 empty unless a guide tells you to add something. Quotes are supported."
            }
            Field::VllmGpuMem => {
                "Fraction of GPU memory vLLM may use. 0.90 leaves some headroom for your desktop. \
                 Lower it if other apps need the GPU."
            }
            Field::VllmMaxSeqs => {
                "How many requests vLLM handles at once. 16 is plenty for one person."
            }
            Field::VllmBatchTokens => {
                "Maximum tokens vLLM processes in one batch. Higher can improve throughput but \
                 uses more memory."
            }
        }
    }

    pub fn recommendation(self) -> &'static str {
        match self {
            Field::Preset => "Balanced",
            Field::Context => "Auto. 8K-16K for chat, 32K+ for long documents.",
            Field::MaxTokens => "Auto, or 1024-2048 for quick chat.",
            Field::Temperature => "0.8 chat, 0.2-0.6 for code.",
            Field::TopP => "0.95",
            Field::RepeatPenalty => "1.1 (1.0 disables it)",
            Field::Effort => "Off for normal models. Medium/High only for thinking models.",
            Field::ReasoningBudget => "-1 (unlimited). Try 512-2048 to speed up thinking.",
            Field::GpuLayers => "Auto. Use All if the whole model fits in VRAM.",
            Field::Batch => "2048",
            Field::Ubatch => "512",
            Field::Threads => "Auto",
            Field::Parallel => "1",
            Field::FlashAttn => "Auto",
            Field::Jinja => "On",
            Field::Mlock => "Off",
            Field::Seed => "Auto (random)",
            Field::SpecType => "Off, unless you know your model supports speculation.",
            Field::DraftModel => "None",
            Field::DraftMax => "3",
            Field::DraftMin => "0",
            Field::Host => "127.0.0.1",
            Field::Port => "8080",
            Field::ExtraArgs => "Empty",
            Field::VllmGpuMem => "0.90",
            Field::VllmMaxSeqs => "16",
            Field::VllmBatchTokens => "8192",
        }
    }
}

const LLAMA_BASICS: [Field; 8] = [
    Field::Preset,
    Field::Context,
    Field::MaxTokens,
    Field::Temperature,
    Field::TopP,
    Field::RepeatPenalty,
    Field::Effort,
    Field::ReasoningBudget,
];

const LLAMA_PERFORMANCE: [Field; 13] = [
    Field::GpuLayers,
    Field::Batch,
    Field::Ubatch,
    Field::Threads,
    Field::Parallel,
    Field::FlashAttn,
    Field::Jinja,
    Field::Mlock,
    Field::Seed,
    Field::SpecType,
    Field::DraftModel,
    Field::DraftMax,
    Field::DraftMin,
];

const LLAMA_ADVANCED: [Field; 3] = [Field::Host, Field::Port, Field::ExtraArgs];

const VLLM_BASICS: [Field; 7] = [
    Field::Preset,
    Field::Context,
    Field::MaxTokens,
    Field::Temperature,
    Field::TopP,
    Field::RepeatPenalty,
    Field::Effort,
];

const VLLM_PERFORMANCE: [Field; 7] = [
    Field::VllmGpuMem,
    Field::VllmMaxSeqs,
    Field::VllmBatchTokens,
    Field::SpecType,
    Field::DraftModel,
    Field::DraftMax,
    Field::Seed,
];

const VLLM_ADVANCED: [Field; 3] = [Field::Host, Field::Port, Field::ExtraArgs];

pub const TABS: [&str; 3] = ["Basics", "Performance", "Advanced"];

pub fn fields_for(backend: Backend, tab: usize) -> &'static [Field] {
    match (backend, tab) {
        (Backend::LlamaCpp, 0) => &LLAMA_BASICS,
        (Backend::LlamaCpp, 1) => &LLAMA_PERFORMANCE,
        (Backend::LlamaCpp, _) => &LLAMA_ADVANCED,
        (Backend::Vllm, 0) => &VLLM_BASICS,
        (Backend::Vllm, 1) => &VLLM_PERFORMANCE,
        (Backend::Vllm, _) => &VLLM_ADVANCED,
    }
}
