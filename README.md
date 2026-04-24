# autoresearch_folktales

> An adaptation of [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for training a small LLM on folktales dataset, running on Apple Silicon Mac.

---

**Watch the full walkthrough on YouTube:** [https://www.youtube.com/watch?v=XXR0zZ0_16M](https://www.youtube.com/watch?v=XXR0zZ0_16M)

---

## About the original project

[autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy is an autonomous AI research setup where an AI agent is given a small but real LLM training environment and let loose to experiment overnight. The agent modifies training code, runs 5-minute experiments, checks whether the result improved, keeps or discards changes, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat).

> *One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. — @karpathy, March 2026*

## Overview

This repo adapts autoresearch to:

1. **Train on a small story dataset** — [folk-mythology-tales](https://huggingface.co/datasets/merve/folk-mythology-tales) dataset from Hugging Face, instead of the original dataset that Karpathy used (which is much larger).
2. **Run on Apple Silicon Mac** — adds MPS (Metal Performance Shaders) support so you can run autonomous LLM research experiments directly on an M1/M2/M3/M4 MacBook or Mac, without needing an NVIDIA GPU.

The core autoresearch loop is preserved: the AI agent iteratively modifies `train.py`, runs 5-minute training experiments, and tracks `val_bpb` (validation bits per byte) to find the best hyperparameters.

## How it works

The repo has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads the folklore training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc.
- **`program.md`** — baseline instructions for the agent. Point your agent here and let it go.

The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared. Each experiment runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation).

## Quick start

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4 with Metal/MPS support), Python 3.10+, [uv](https://docs.astral.sh/uv/), and an AI coding agent such as [Claude Code](https://claude.ai/code) or [Mistral Vibe](https://mistral.ai).

```bash
# 1. Clone this repo
git clone https://github.com/thu-vu92/autoresearch_folktales.git
cd autoresearch_folktales

# 2. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync

# 4. Download folklore data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 5. Run a single training experiment manually (~5 min)
uv run train.py
```

If the above commands all work, your setup is ready. You can now run the agent in autonomous research mode.

## Running the agent

Spin up Claude Code (or any coding agent) in this repo, then prompt:

```
Have a look at program.md and let's kick off a new experiment!
```

The `program.md` file is a lightweight "skill" that tells the agent how to run experiments, record results, and iterate.

> **Note on Apple Silicon performance:** MPS is significantly slower than a modern NVIDIA GPU for this workload — expect ~85–90 optimizer steps in 5 minutes vs. 1000s on a fast GPU. This means hyperparameter improvements found by the agent are real but may require more steps to fully manifest. The agent still runs and iterates correctly; results are just compute-limited at this scale.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Dataset

This project uses the [merve/folk-mythology-tales](https://huggingface.co/datasets/merve/folk-mythology-tales) dataset from Hugging Face — a collection of folk and mythology tales. `prepare.py` handles the download and tokenization automatically.

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes of wall clock time, making experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc.).
- **Apple Silicon support.** FlashAttention-3 dependency is removed; falls back to PyTorch's native SDPA with manual sliding window causal masking. `torch.compile` paths unsupported on MPS are disabled. Optimizer states are precisely cast for Metal compatibility.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs.

## Credits

- Original [autoresearch](https://github.com/karpathy/autoresearch) by [Andrej Karpathy](https://github.com/karpathy)
- macOS/MPS adaptation by [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- Folklore dataset: [merve/folk-mythology-tales](https://huggingface.co/datasets/merve/folk-mythology-tales)

## License

MIT
