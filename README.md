# autoresearch_folktales

> An adaptation of [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for training a small LLM on a folktales dataset, configured for Windows + NVIDIA GTX 1650.

## About the original project

[autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy is an autonomous AI research setup where an AI agent is given a small but real LLM training environment and let loose to experiment overnight. The agent modifies training code, runs 5-minute experiments, checks whether the result improved, keeps or discards changes, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat).

> *One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. — @karpathy, March 2026*

## Overview

This repo adapts autoresearch to:

1. **Train on a small story dataset** — [folk-mythology-tales](https://huggingface.co/datasets/merve/folk-mythology-tales) dataset from Hugging Face, instead of the original dataset that Karpathy used (which is much larger).
2. **Run on Windows + CUDA GPU** — tested with Conda environment `folktales` on NVIDIA GeForce GTX 1650 using fp16 autocast and small VRAM-safe defaults.

The core autoresearch loop is preserved: the AI agent iteratively modifies `train.py`, runs 5-minute training experiments, and tracks `val_bpb` (validation bits per byte) to find the best hyperparameters.

## How it works

The repo has three files that matter:

- **`prepare.py`** — one-time data prep (downloads training data, trains tokenizer) plus runtime utilities (dataloader, evaluation).
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc.
- **`program.md`** — baseline instructions for the agent. Point your agent here and let it go.

The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared. Each experiment runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation).


```powershell
# 1. Clone this repo
git clone https://github.com/Abhaysharma53/test_autoresearch_folktales.git
cd test_autoresearch_folktales

# 2. Activate conda env
conda activate folktales

# 3. Download data and train tokenizer (one-time)
python prepare.py

# 4. Run a single training experiment (~5 min)
python train.py
```

If the above commands all work, your setup is ready. You can now run the agent in autonomous research mode.

## Running the agent

Spin up Claude Code (or any coding agent) in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is a lightweight "skill" that tells the agent how to run experiments, record results, and iterate.

> **Current setup note:** this repo currently targets Windows + GTX 1650. The training loop includes fp16 stability guards and smaller defaults to avoid mid-run NaN crashes on 4GB-class GPUs.

## Why `agent_loop.py` is used

`agent_loop.py` is the automation runner used in this repo to execute the experiment campaign end-to-end without manual intervention.

- **What it does:** runs a fixed sequence of experiments, launches `python train.py`, streams progress to terminal, parses final metrics from `run.log`, and appends structured rows to `results.tsv`.
- **Why we use it:** avoids repetitive manual run/parse/log steps and enforces a strict stop condition (exactly 6 experiments in this setup).
- **How it handles interruptions:** supports resume behavior by checking how many rows already exist in `results.tsv` and continuing from the next pending experiment.
- **How it decides status:** marks each run as `keep`, `discard`, or `crash` based on metric validity and best-so-far comparison.

Run it with:

```powershell
python agent_loop.py
```

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
agent_loop.py   — deterministic multi-experiment runner + results logging
pyproject.toml  — dependencies
```

## Dataset

This project uses the [merve/folk-mythology-tales](https://huggingface.co/datasets/merve/folk-mythology-tales) dataset from Hugging Face — a collection of folk and mythology tales. `prepare.py` handles the download and tokenization automatically.

## Design choices

- **Single primary file to modify.** Most experiments touch `train.py`, while `prepare.py` and `program.md` may be updated for environment compatibility and process clarity.
- **Fixed time budget.** Training always runs for exactly 5 minutes of wall clock time, making experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc.).
- **Windows CUDA stability.** GTX 1650-compatible fp16 autocast, conservative defaults, and non-finite loss/gradient guards help keep long runs stable.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs.

## Credits

- Original [autoresearch](https://github.com/karpathy/autoresearch) by [Andrej Karpathy](https://github.com/karpathy)
- Folklore dataset: [merve/folk-mythology-tales](https://huggingface.co/datasets/merve/folk-mythology-tales)

