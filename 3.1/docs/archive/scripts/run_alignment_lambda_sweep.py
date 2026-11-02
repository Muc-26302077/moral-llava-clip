"""Run a validation-only LLaVA alignment lambda sweep."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "outputs/alignment_lambda_sweep_status.json"
LOG_ROOT = ROOT / "outputs/alignment_lambda_sweep_logs"
RUNS = (
    ("005", "0.05"),
    ("010", "0.1"),
    ("020", "0.2"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_status(status: dict) -> None:
    status["updated_at"] = now()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run_phase(status: dict, name: str, command: list[str], environment: dict[str, str]) -> None:
    log_path = LOG_ROOT / f"{name}.log"
    phase = {
        "name": name, "status": "RUNNING", "command": command,
        "started_at": now(), "log": str(log_path),
    }
    status["phases"].append(phase)
    status["current_phase"] = name
    save_status(status)
    print(f"{now()} Starting {name}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        phase["pid"] = child.pid
        save_status(status)
        code = child.wait()
    phase.update({"exit_code": code, "finished_at": now(), "status": "PASS" if code == 0 else "FAIL"})
    save_status(status)
    if code:
        raise RuntimeError(f"{name} exited {code}; see {log_path}")


def main() -> None:
    os.chdir(ROOT)
    run_dirs = [ROOT / f"runs/alignment-moral-lambda-{label}-b16" for label, _ in RUNS]
    result_paths = [ROOT / f"outputs/alignment_moral_lambda_{label}_b16_validation.json" for label, _ in RUNS]
    result_summary = ROOT / "outputs/alignment_lambda_sweep_results.json"
    if STATUS.exists() or LOG_ROOT.exists() or result_summary.exists() or any(path.exists() for path in run_dirs + result_paths):
        raise FileExistsError("Sweep outputs already exist; inspect them before launching another run")

    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    LOG_ROOT.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "RUNNING", "pid": os.getpid(), "started_at": now(), "phases": [],
        "protocol": "train/validation only; same seed and settings; validation paired R@1 selects checkpoint; no test evaluation",
        "lambdas": [float(weight) for _, weight in RUNS],
        "results": [],
    }
    save_status(status)

    try:
        for label, weight in RUNS:
            run_name = f"alignment-moral-lambda-{label}-b16"
            output_dir = ROOT / "runs" / run_name
            run_phase(status, f"train_lambda_{label}", [
                sys.executable, "-u", "-m", "moral_llava.train_alignment",
                "--vision-model", "/root/autodl-tmp/clip-vit-base-patch16",
                "--output-dir", str(output_dir), "--batch-size", "64", "--epochs", "10",
                "--moral-weight", weight, "--moral-similarity", "paper_scaled",
                "--selection-metric", "paired_r1", "--seed", "42", "--text-max-length", "320",
            ], environment)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            pointer = json.loads((output_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
            if (
                summary.get("status") != "PASS"
                or summary.get("run_kind") != "training"
                or summary.get("optimizer_steps") != 330
                or summary.get("selection_metric") != "paired_r1"
                or summary.get("test_evaluated") is not False
                or pointer.get("split") != "validation"
                or pointer.get("selection_metric") != "paired_r1"
            ):
                raise RuntimeError(f"Unexpected completed run metadata for {run_name}: {summary} {pointer}")

            epoch = int(Path(pointer["checkpoint"]).name.split("-")[-1])
            validation_path = output_dir / f"validation_epoch_{epoch:03d}.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            metrics = validation["metrics"]
            row = {
                "lambda": float(weight), "run": run_name, "checkpoint": pointer["checkpoint"],
                "selected_epoch": epoch,
                "paired_r1": metrics["same_image_pairing"]["r_at_1"],
                "moral_map": metrics["label_relevance"]["map_i2t"],
                "cross_image_moral_map": metrics["cross_image_moral_relevance"]["map_i2t"],
                "validation_record": str(validation_path),
                "test_evaluated": False,
            }
            status["results"].append(row)
            save_status(status)

        result_summary.write_text(json.dumps(status["results"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status.update({"status": "PASS", "finished_at": now(), "results_path": str(result_summary)})
    except BaseException as error:
        status.update({
            "status": "FAIL", "finished_at": now(), "error": str(error),
            "traceback": traceback.format_exc(),
        })
        save_status(status)
        raise

    save_status(status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
