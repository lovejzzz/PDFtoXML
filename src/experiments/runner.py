"""Autoresearch experiment runner.

Orchestrates bounded experiments following the autoresearch pattern:
1. Define a search space of hyperparameter configs
2. Run each experiment with fixed eval harness
3. Log results to results.tsv
4. Keep/discard based on dev score improvement
5. Commit kept experiments, revert discarded ones

Usage:
    python -m src.experiments.runner                    # run all pending experiments
    python -m src.experiments.runner --list             # list experiment configs
    python -m src.experiments.runner --run exp_name     # run a specific experiment
    python -m src.experiments.runner --dashboard        # show results dashboard
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results.tsv")
EXPERIMENTS_LOG = os.path.join(PROJECT_ROOT, "outputs", "experiments_log.json")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Minimum score improvement to keep an experiment
KEEP_THRESHOLD = 0.005

# Current best score (updated dynamically from results.tsv)
def _get_best_score() -> float:
    """Read the best non-roundtrip score from results.tsv."""
    if not os.path.exists(RESULTS_PATH):
        return 0.0
    best = 0.0
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.startswith("commit") or line.startswith("roundtrip"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    score = float(parts[1])
                    if score > best:
                        best = score
                except ValueError:
                    pass
    return best


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    name: str
    description: str
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_ff: int = 512
    dropout: float = 0.1
    img_height: int = 512
    img_width: int = 384
    max_seq_len: int = 1400
    eval_every: int = 5
    patience: int = 20
    use_synthetic: int = 1
    # Extra flags
    label_smoothing: float = 0.0
    warmup_steps: int = 0

    def to_cli_args(self) -> list[str]:
        """Convert to CLI argument strings."""
        args = []
        for k, v in asdict(self).items():
            if k in ("name", "description"):
                continue
            args.append(f"{k}={v}")
        return args


# ============================================================
# EXPERIMENT SEARCH SPACE
# Each config varies ONE focused thing from the current best.
# ============================================================

EXPERIMENT_CONFIGS = [
    # --- Model architecture ---
    ExperimentConfig(
        name="wider_d512",
        description="Double model width to d=512",
        d_model=512, dim_ff=1024, nhead=8,
        epochs=40, batch_size=2, lr=5e-5,
    ),
    ExperimentConfig(
        name="deeper_L6",
        description="6 decoder layers instead of 4",
        num_layers=6, epochs=40, lr=8e-5,
    ),
    ExperimentConfig(
        name="shallow_L2",
        description="Lighter model: 2 decoder layers",
        num_layers=2, epochs=50, lr=2e-4,
    ),

    # --- Image resolution ---
    ExperimentConfig(
        name="highres_768x576",
        description="Higher resolution input images",
        img_height=768, img_width=576,
        batch_size=2, epochs=40, lr=8e-5,
    ),
    ExperimentConfig(
        name="lowres_256x192",
        description="Lower resolution (faster training)",
        img_height=256, img_width=192,
        batch_size=8, epochs=60, lr=2e-4,
    ),

    # --- Learning rate ---
    ExperimentConfig(
        name="lr_3e4",
        description="Higher learning rate 3e-4",
        lr=3e-4, epochs=40,
    ),
    ExperimentConfig(
        name="lr_5e5",
        description="Lower learning rate 5e-5",
        lr=5e-5, epochs=60,
    ),

    # --- Regularization ---
    ExperimentConfig(
        name="dropout_02",
        description="Higher dropout 0.2",
        dropout=0.2, epochs=50,
    ),
    ExperimentConfig(
        name="label_smooth_01",
        description="Label smoothing 0.1",
        label_smoothing=0.1, epochs=50,
    ),

    # --- Training strategy ---
    ExperimentConfig(
        name="real_only_long",
        description="Real data only, longer training, higher LR",
        use_synthetic=0, epochs=80, lr=3e-4, batch_size=2,
        patience=30, eval_every=5,
    ),
    ExperimentConfig(
        name="synth_heavy_augment",
        description="Synthetic + strong online augmentation",
        use_synthetic=1, epochs=60, lr=1e-4,
        dropout=0.15,
    ),

    # --- Batch size ---
    ExperimentConfig(
        name="batch_8",
        description="Larger batch size 8",
        batch_size=8, lr=2e-4, epochs=50,
    ),
]


def _load_experiments_log() -> dict:
    """Load the experiments log."""
    if os.path.exists(EXPERIMENTS_LOG):
        with open(EXPERIMENTS_LOG) as f:
            return json.load(f)
    return {"experiments": []}


def _save_experiments_log(log: dict):
    """Save the experiments log."""
    os.makedirs(os.path.dirname(EXPERIMENTS_LOG), exist_ok=True)
    with open(EXPERIMENTS_LOG, "w") as f:
        json.dump(log, f, indent=2)


def _get_completed_experiments() -> set[str]:
    """Get names of already-completed experiments."""
    log = _load_experiments_log()
    return {e["name"] for e in log["experiments"]}


def run_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config.name}")
    print(f"  {config.description}")
    print(f"{'='*60}\n")

    best_before = _get_best_score()
    print(f"  Current best score: {best_before:.4f}")

    # Build CLI args
    args = config.to_cli_args()
    cmd = [sys.executable, "-m", "src.experiments.train"] + args

    print(f"  Command: {' '.join(cmd)}")
    start_time = time.time()

    # Run training
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=3600,  # 1 hour max per experiment
    )

    elapsed = time.time() - start_time

    # Parse output for the score
    output = result.stdout + result.stderr
    score = 0.0
    event_f1 = 0.0
    pitch_acc = 0.0
    rhythm_acc = 0.0

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("score="):
            try:
                score = float(line.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif "score=" in line and "event_f1=" in line:
            # Aggregate line
            for part in line.split():
                if part.startswith("score="):
                    try:
                        score = float(part.split("=")[1])
                    except ValueError:
                        pass
                elif part.startswith("event_f1="):
                    try:
                        event_f1 = float(part.split("=")[1])
                    except ValueError:
                        pass
                elif part.startswith("pitch_acc="):
                    try:
                        pitch_acc = float(part.split("=")[1])
                    except ValueError:
                        pass
                elif part.startswith("rhythm_acc="):
                    try:
                        rhythm_acc = float(part.split("=")[1])
                    except ValueError:
                        pass

    # Determine keep/discard
    improvement = score - best_before
    status = "keep" if improvement >= KEEP_THRESHOLD else "discard"
    if result.returncode != 0 and score == 0.0:
        status = "crash"

    exp_result = {
        "name": config.name,
        "description": config.description,
        "score": score,
        "event_f1": event_f1,
        "pitch_acc": pitch_acc,
        "rhythm_acc": rhythm_acc,
        "improvement": improvement,
        "status": status,
        "elapsed_seconds": int(elapsed),
        "best_before": best_before,
        "returncode": result.returncode,
        "config": asdict(config),
    }

    # Log
    log = _load_experiments_log()
    log["experiments"].append(exp_result)
    _save_experiments_log(log)

    # Print summary
    print(f"\n  RESULT: {config.name}")
    print(f"    Score: {score:.4f} (was {best_before:.4f}, Δ={improvement:+.4f})")
    print(f"    Status: {status.upper()}")
    print(f"    Time: {int(elapsed)}s")

    if status == "keep":
        print(f"    ✓ KEPT — score improved by {improvement:.4f}")
        # Save the best checkpoint with experiment name
        best_ckpt = os.path.join(CHECKPOINTS_DIR, "best.pt")
        named_ckpt = os.path.join(CHECKPOINTS_DIR, f"best_{config.name}.pt")
        if os.path.exists(best_ckpt):
            import shutil
            shutil.copy2(best_ckpt, named_ckpt)
    elif status == "discard":
        print(f"    ✗ DISCARDED — improvement {improvement:+.4f} below threshold {KEEP_THRESHOLD}")
    else:
        print(f"    ✗ CRASHED — returncode {result.returncode}")
        if result.stderr:
            print(f"    stderr: {result.stderr[-500:]}")

    return exp_result


def run_all_pending():
    """Run all experiments that haven't been completed yet."""
    completed = _get_completed_experiments()
    pending = [c for c in EXPERIMENT_CONFIGS if c.name not in completed]

    if not pending:
        print("All experiments completed!")
        print_dashboard()
        return

    print(f"\n{len(pending)} experiments pending, {len(completed)} completed")
    print(f"Pending: {[c.name for c in pending]}\n")

    for i, config in enumerate(pending):
        print(f"\n--- Experiment {i+1}/{len(pending)} ---")
        try:
            run_experiment(config)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {config.name} exceeded 1 hour")
            log = _load_experiments_log()
            log["experiments"].append({
                "name": config.name,
                "description": config.description,
                "score": 0.0,
                "status": "crash",
                "elapsed_seconds": 3600,
            })
            _save_experiments_log(log)
        except Exception as e:
            print(f"  ERROR: {config.name}: {e}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print_dashboard()


def print_dashboard():
    """Print a summary dashboard of all experiments."""
    log = _load_experiments_log()
    experiments = log.get("experiments", [])

    if not experiments:
        print("No experiments logged yet.")
        return

    print(f"\n{'='*80}")
    print(f"{'EXPERIMENT DASHBOARD':^80}")
    print(f"{'='*80}")
    print(f"{'Name':<25} {'Score':>7} {'Δ':>8} {'Status':<10} {'Time':>6} Description")
    print(f"{'-'*80}")

    # Sort by score descending
    sorted_exps = sorted(experiments, key=lambda e: e.get("score", 0), reverse=True)

    for exp in sorted_exps:
        name = exp.get("name", "?")[:24]
        score = exp.get("score", 0)
        improvement = exp.get("improvement", 0)
        status = exp.get("status", "?")
        elapsed = exp.get("elapsed_seconds", 0)
        desc = exp.get("description", "")[:30]

        status_marker = {"keep": "KEEP", "discard": "DISCARD", "crash": "CRASH"}.get(status, "?")
        time_str = f"{elapsed//60}m{elapsed%60:02d}s" if elapsed else "?"

        print(f"  {name:<24} {score:>6.4f} {improvement:>+7.4f} {status_marker:<10} {time_str:>6} {desc}")

    print(f"{'-'*80}")

    # Summary stats
    kept = [e for e in experiments if e.get("status") == "keep"]
    discarded = [e for e in experiments if e.get("status") == "discard"]
    crashed = [e for e in experiments if e.get("status") == "crash"]
    best = max(experiments, key=lambda e: e.get("score", 0))

    print(f"  Total: {len(experiments)} | Kept: {len(kept)} | "
          f"Discarded: {len(discarded)} | Crashed: {len(crashed)}")
    print(f"  Best: {best.get('name', '?')} ({best.get('score', 0):.4f})")
    print(f"{'='*80}\n")


def list_configs():
    """List all experiment configs."""
    completed = _get_completed_experiments()
    print(f"\nExperiment Configs ({len(EXPERIMENT_CONFIGS)} total):\n")
    for c in EXPERIMENT_CONFIGS:
        status = "DONE" if c.name in completed else "PENDING"
        print(f"  [{status:>7}] {c.name:<25} {c.description}")


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_configs()
    elif "--dashboard" in sys.argv:
        print_dashboard()
    elif "--run" in sys.argv:
        idx = sys.argv.index("--run") + 1
        if idx < len(sys.argv):
            name = sys.argv[idx]
            config = next((c for c in EXPERIMENT_CONFIGS if c.name == name), None)
            if config:
                run_experiment(config)
            else:
                print(f"Unknown experiment: {name}")
                list_configs()
        else:
            print("Specify experiment name: --run <name>")
    else:
        run_all_pending()
