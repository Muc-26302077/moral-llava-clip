"""Run the two fixed B/16 alignment experiments, then evaluate validation-best weights."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "outputs/alignment_experiment_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_status(status: dict) -> None:
    status["updated_at"] = now()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main() -> None:
    os.chdir(ROOT)
    runs = {"clip": ROOT / "runs/alignment-clip-b16", "moral": ROOT / "runs/alignment-moral-b16"}
    # 旧结果不覆盖，断点恢复也不能靠重复执行来冒充。
    if STATUS.exists() or any(path.exists() for path in runs.values()):
        raise FileExistsError("Formal run or status already exists; inspect it before launching another run")
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_root = ROOT / "outputs/alignment_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    status = {"status": "RUNNING", "pid": os.getpid(), "started_at": now(), "phases": [],
              "protocol": "same seed/config; only moral_weight differs; validation moral MAP selects checkpoint; test after both runs"}
    save_status(status)

    def run_phase(name: str, command: list[str]) -> None:
        phase = {"name": name, "status": "RUNNING", "command": command,
                 "started_at": now(), "log": str(log_root / f"{name}.log")}
        status["phases"].append(phase)
        status["current_phase"] = name
        save_status(status)
        print(f"{now()} Starting {name}", flush=True)
        with Path(phase["log"]).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            phase["pid"] = child.pid
            save_status(status)
            code = child.wait()
        phase.update({"exit_code": code, "finished_at": now(), "status": "PASS" if code == 0 else "FAIL"})
        save_status(status)
        if code:
            raise RuntimeError(f"{name} exited {code}; see {phase['log']}")

    try:
        for name, weight in (("clip", "0"), ("moral", "0.4")):
            run_phase(f"train_{name}", [
                sys.executable, "-u", "-m", "moral_llava.train_alignment",
                "--vision-model", "/root/autodl-tmp/clip-vit-base-patch16",
                "--output-dir", str(runs[name]), "--batch-size", "64", "--epochs", "10",
                "--moral-weight", weight, "--moral-similarity", "paper_scaled", "--seed", "42",
                "--text-max-length", "320",
            ])
            summary = json.loads((runs[name] / "summary.json").read_text())
            if summary["status"] != "PASS" or summary["run_kind"] != "training" or summary["optimizer_steps"] != 330:
                raise RuntimeError(f"Unexpected training completion: {summary}")

        # 两组均结束后才接触 test，不根据 test 分数挑 epoch。
        checkpoints = {}
        for name in runs:
            pointer = json.loads((runs[name] / "best_checkpoint.json").read_text())
            if pointer["split"] != "validation" or pointer["selection_metric"] != "moral_map":
                raise RuntimeError(f"Unexpected checkpoint selection: {pointer}")
            checkpoints[name] = pointer["checkpoint"]
            run_phase(f"test_{name}", [
                sys.executable, "-u", "scripts/evaluate_alignment.py",
                "--checkpoint", checkpoints[name], "--split", "test", "--precision", "bf16",
                "--text-max-length", "320",
                "--output", f"outputs/alignment_{name}_b16_test.json",
            ])
        run_phase("retrieve_user_image", [
            sys.executable, "-u", "scripts/retrieve_alignment.py",
            "--checkpoint", checkpoints["moral"], "--image", "uploads/user_test_20261018.png",
            "--top-k", "5", "--text-max-length", "320",
            "--output", "outputs/alignment_moral_b16_user_retrieval.json",
        ])
        status.update({"status": "PASS", "finished_at": now(), "selected_checkpoints": checkpoints})
    except BaseException as error:
        status.update({"status": "FAIL", "finished_at": now(), "error": str(error), "traceback": traceback.format_exc()})
        save_status(status)
        raise
    save_status(status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
