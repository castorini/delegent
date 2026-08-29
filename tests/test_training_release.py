from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_config_and_lineage_are_consistent() -> None:
    config = json.loads((ROOT / "configs/gemma4-sft.json").read_text())
    lineage = json.loads((ROOT / "results/training_lineage.json").read_text())
    assert config["base_model"] == lineage["reported_adapter"]["base_model"]
    assert config["base_model_revision"] == lineage["reported_adapter"]["base_model_revision"]
    assert config["lora_rank"] == lineage["operative_lora"]["rank"] == 16
    assert config["lora_alpha"] == lineage["operative_lora"]["alpha"] == 32
    assert config["max_steps"] == lineage["final_continuation_stage"]["optimizer_steps"]


def test_training_dry_run_needs_no_ml_dependencies(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/train_retrieval_sft.py"),
        "--config",
        str(ROOT / "configs/gemma4-sft.json"),
        "--train-files",
        str(ROOT / "examples/training_example.jsonl"),
        "--validation-files",
        str(ROOT / "examples/training_example.jsonl"),
        "--output-dir",
        str(tmp_path),
        "--dry-run",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["status"] == "validated"
    assert payload["train_rows"] == payload["validation_rows"] == 1
    assert payload["model_or_adapter_loaded"] is False


def test_historical_training_sources_match_lineage() -> None:
    lineage = json.loads((ROOT / "results/training_lineage.json").read_text())
    for record in lineage["historical_training_code"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
