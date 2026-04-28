"""
Deterministic autoresearch loop for exactly 6 experiments.
Prints live progress in terminal and appends each result to results.tsv.
"""

import re
import subprocess
from pathlib import Path

TRAIN = Path("train.py")
RESULTS = Path("results.tsv")
RUN_LOG = Path("run.log")


def get_head_hash():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def ensure_results_header():
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
    if not RESULTS.exists():
        RESULTS.write_text(header, encoding="utf-8")
        return
    content = RESULTS.read_text(encoding="utf-8")
    if not content.startswith("commit\tval_bpb\tmemory_gb\tstatus\tdescription"):
        RESULTS.write_text(header, encoding="utf-8")


def completed_experiments_count():
    if not RESULTS.exists():
        return 0
    lines = [ln for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Exclude header
    return max(0, len(lines) - 1)


def patch_train_hparams(source, updates):
    patched = source
    for key, value in updates.items():
        pattern = rf"(?m)^({re.escape(key)}\s*=\s*)(.+?)(\s*#.*)?$"
        replacement = rf"\g<1>{value}\3"
        patched, n = re.subn(pattern, replacement, patched, count=1)
        if n == 0:
            raise RuntimeError(f"Could not patch hyperparameter: {key}")
    return patched


def run_training_with_live_output():
    proc = subprocess.Popen(
        ["python", "train.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with RUN_LOG.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
    return proc.wait()


def parse_metrics():
    text = RUN_LOG.read_text(encoding="utf-8", errors="ignore")
    m_bpb = re.search(r"^val_bpb:\s*([0-9.]+)", text, re.MULTILINE)
    m_vram = re.search(r"^peak_vram_mb:\s*([0-9.]+)", text, re.MULTILINE)
    if not m_bpb:
        return None, 0.0
    val_bpb = float(m_bpb.group(1))
    peak_mb = float(m_vram.group(1)) if m_vram else 0.0
    return val_bpb, round(peak_mb / 1024.0, 1)


def append_result(commit, val_bpb, memory_gb, status, description):
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def main():
    ensure_results_header()
    baseline_train = TRAIN.read_text(encoding="utf-8")
    head_hash = get_head_hash()
    best_bpb = None

    experiments = [
        ("baseline", {}),
        ("embedding_lr=0.9", {"EMBEDDING_LR": "0.9"}),
        ("matrix_lr=0.025", {"MATRIX_LR": "0.025"}),
        ("warmdown_ratio=0.4", {"WARMDOWN_RATIO": "0.4"}),
        ("depth=3 batch=2", {"DEPTH": "3", "DEVICE_BATCH_SIZE": "2", "TOTAL_BATCH_SIZE": "2**10"}),
        ("embedding_lr=1.0", {"EMBEDDING_LR": "1.0"}),
    ]

    already_done = completed_experiments_count()
    if already_done >= 6:
        print("Six experiments already completed. Nothing to run.")
        return

    print(f"Starting deterministic 6-experiment loop (resume from {already_done + 1}/6)...")
    for idx, (description, updates) in enumerate(experiments[already_done:], start=already_done + 1):
        print(f"\n{'='*72}")
        print(f"Experiment {idx}/6: {description}")
        print(f"{'='*72}")

        train_text = patch_train_hparams(baseline_train, updates)
        TRAIN.write_text(train_text, encoding="utf-8")

        code = run_training_with_live_output()
        val_bpb, memory_gb = parse_metrics()

        if code != 0 or val_bpb is None:
            status = "crash"
            val_bpb = 0.0
            memory_gb = 0.0
            print(f"[exp {idx}] CRASH")
        else:
            if best_bpb is None or val_bpb < best_bpb:
                status = "keep"
                best_bpb = val_bpb
            else:
                status = "discard"
            print(f"[exp {idx}] val_bpb={val_bpb:.6f}, memory_gb={memory_gb:.1f}, status={status}")

        append_result(head_hash, val_bpb, memory_gb, status, description)

    TRAIN.write_text(baseline_train, encoding="utf-8")
    print("\nLoop complete. Stopped automatically after 6 experiments.")
    print(f"Best val_bpb: {best_bpb if best_bpb is not None else 'n/a'}")
    print(f"Results saved to: {RESULTS}")


if __name__ == "__main__":
    main()