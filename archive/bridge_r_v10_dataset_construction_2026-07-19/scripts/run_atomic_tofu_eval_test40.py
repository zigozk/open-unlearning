from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from atomic_tofu.eval_extension import mock_slot_analysis
from atomic_tofu.io import sha256_json, write_json, write_jsonl
from atomic_tofu.pipeline import run_units

BASE = Path("data/atomic_tofu/v2.0-rc1")
ROOT = BASE / "api" / "eval_extension" / "test-40"
INPUT_UNITS = BASE / "api" / "eval_extension" / "input_units.jsonl"
SOURCE = BASE / "source" / "tofu_full.jsonl"
OFFICIAL = BASE / "official_anchors" / "official_eval_sidecars.jsonl"

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main() -> None:
    if ROOT.exists():
        raise SystemExit(f"refusing to reuse existing output directory: {ROOT}")
    source_by_id = {row["qa_id"]: row for row in read_jsonl(SOURCE)}
    existing_units = read_jsonl(INPUT_UNITS)
    official_ids = {row["qa_id"] for row in read_jsonl(OFFICIAL)}
    calibration_ids: set[str] = set()
    for path in (BASE / "api/eval_extension/calibration/generated/generated_60_manifest.json", BASE / "api/eval_extension/anchor_calibration/anchor_20_manifest.json"):
        calibration_ids.update(row["qa_id"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"])
    eligible = sorted(row for row in existing_units if row["unit_id"] not in official_ids and row["unit_id"] not in calibration_ids)
    selected = eligible[:40]
    if [row["unit_id"] for row in selected] != [f"tofu_full_{i:04d}" for i in range(400, 440)]:
        raise SystemExit("deterministic selection did not resolve to tofu_full_0400..0439")
    (ROOT / "source").mkdir(parents=True)
    (ROOT / "official_anchors").mkdir(parents=True)
    (ROOT / "audit").mkdir(parents=True)
    (ROOT / "source/tofu_full.jsonl").symlink_to(SOURCE.resolve())
    for name in ("official_alignment.json", "official_eval_sidecars.jsonl"):
        (ROOT / "official_anchors" / name).symlink_to((BASE / "official_anchors" / name).resolve())
    units, manifest_rows = [], []
    for row in selected:
        qa_id = row["unit_id"]
        source_row = source_by_id[qa_id]
        source_payload = {key: source_row[key] for key in ("qa_id", "question", "answer")}
        source_sha256 = sha256_json(source_payload)
        payload = {**source_payload, "slot_analysis": mock_slot_analysis({"payload": source_payload})}
        input_sha256 = sha256_json(payload)
        units.append({"unit_id": qa_id, "content_sha256": input_sha256, "payload": payload, "perturbation_count": 5, "source_sha256": source_sha256})
        manifest_rows.append({"qa_id": qa_id, "source_sha256": source_sha256, "input_sha256": input_sha256, "question": source_row["question"], "answer": source_row["answer"], "source_or_generated": "generated_candidate"})
    api_dir = ROOT / "api/eval_extension"
    api_dir.mkdir(parents=True)
    write_jsonl(api_dir / "input_units.jsonl", units)
    write_json(ROOT / "manifest.json", {"selection_rule": "sorted existing non-official Eval input_units by qa_id, excluding official sidecars and existing 60/20 calibration IDs; first 40", "count": 40, "qa_ids": [row["qa_id"] for row in manifest_rows], "rows": manifest_rows, "candidate_only": True, "source_path": str(SOURCE.resolve()), "existing_input_units_path": str(INPUT_UNITS.resolve())})
    if "--prepare-only" in sys.argv:
        print(json.dumps({"status": "prepared", "root": str(ROOT), "manifest": str(ROOT / "manifest.json"), "count": 40}, ensure_ascii=False, indent=2))
        return
    os.environ.update({"ATOMIC_TOFU_MAX_CONCURRENCY": "1", "ATOMIC_TOFU_MAX_VALIDATION_REPAIRS": "1", "ATOMIC_TOFU_EVAL_MODEL": "gpt-5.6-luna", "ATOMIC_TOFU_REASONING_EFFORT": "medium"})
    report = run_units(ROOT, "eval_extension", "openai", resume=False, unit_ids={row["unit_id"] for row in units})
    outputs_path = api_dir / "candidate_outputs.jsonl"
    outputs = read_jsonl(outputs_path) if outputs_path.exists() else []
    by_id = {row["unit_id"]: row for row in units}
    for output in outputs:
        output["source_sha256"] = by_id[output["unit_id"]]["source_sha256"]
    write_jsonl(outputs_path, outputs)
    write_json(ROOT / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
