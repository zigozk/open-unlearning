from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    root = Path("data/atomic_tofu/v2.0-rc1")
    view_root = root / "api" / "eval_extension" / "eval_a_candidate_view"
    view_rows = read_jsonl(view_root / "candidate_outputs_4000_eval_a.jsonl")
    provenance_rows = read_jsonl(view_root / "candidate_outputs_4000_eval_a.provenance.jsonl")
    view_by_id = {row["qa_id"]: row for row in view_rows}
    provenance_by_id = {row["qa_id"]: row for row in provenance_rows}
    request_by_id = {row["request_id"]: row for row in read_jsonl(root / "request_graph" / "requests.jsonl")}

    row_roles: dict[str, set[str]] = defaultdict(set)
    row_bundles: dict[str, set[str]] = defaultdict(set)
    bundle_count_by_family: dict[str, int] = defaultdict(int)
    bundle_ids: list[str] = []
    missing_requests: list[dict[str, str]] = []
    for family in ("pure", "balanced"):
        for path in sorted((root / "bundles" / family).rglob("*.json")):
            if path.name.endswith(".manifest.json"):
                continue
            bundle = json.loads(path.read_text(encoding="utf-8"))
            bundle_id = bundle["bundle_id"]
            bundle_ids.append(bundle_id)
            bundle_count_by_family[family] += 1
            for qa_id in bundle.get("forget_qa_ids", []):
                row_roles[qa_id].add("forget")
                row_bundles[qa_id].add(bundle_id)
            for request_id in bundle.get("request_ids", []):
                request = request_by_id.get(request_id)
                if request is None:
                    missing_requests.append({"bundle_id": bundle_id, "request_id": request_id})
                    continue
                for qa_id in request.get("protected_eval_qa_ids", []):
                    row_roles[qa_id].add("protected_eval")
                    row_bundles[qa_id].add(bundle_id)

    decisions: list[dict[str, Any]] = []
    missing_view_rows: list[str] = []
    for qa_id in sorted(row_roles):
        row = view_by_id.get(qa_id)
        provenance = provenance_by_id.get(qa_id)
        if row is None or provenance is None:
            missing_view_rows.append(qa_id)
            continue
        decisions.append({
            "qa_id": qa_id,
            "bundle_roles": sorted(row_roles[qa_id]),
            "bundle_ids": sorted(row_bundles[qa_id]),
            "source_or_generated": provenance["source_or_generated"],
            "candidate_hash": canonical_hash(row),
            "candidate_hash_algorithm": "sha256(canonical_eval_a_row_json_utf8_sort_keys)",
            "generated_candidate_sha256": provenance.get("generated_candidate_sha256"),
            "official_sidecar_sha256": provenance.get("official_sidecar_sha256"),
            "review_status": "pending_human_review",
            "decision": None,
            "reviewer": None,
            "review_decision_binding": {"candidate_hash": canonical_hash(row), "candidate_only": True, "model_lock": "floating_alias"},
            "mechanical_context": {"official_fields_overwritten": provenance.get("official_fields_overwritten", False), "human_review_status_from_source": provenance.get("human_review_status")},
        })

    output_root = root / "audit" / "bundle_eval_row_review_2026-07-18"
    output_root.mkdir(parents=True, exist_ok=True)
    decisions_path = output_root / "review_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "artifact": "formal bundle Eval-A row review inventory",
        "status": "inventory_persisted_human_review_pending",
        "candidate_only": True,
        "model_lock": "floating_alias",
        "human_review_completed": False,
        "review_decisions_are_hash_bound": True,
        "review_decision_hash_algorithm": "sha256(canonical_eval_a_row_json_utf8_sort_keys)",
        "bundle_count": len(bundle_ids),
        "bundle_count_by_family": dict(bundle_count_by_family),
        "unique_eval_rows": len(row_roles),
        "persisted_review_decisions": len(decisions),
        "generated_rows": sum(provenance_by_id[qa_id]["source_or_generated"] == "generated" for qa_id in row_roles if qa_id in provenance_by_id),
        "official_rows": sum(provenance_by_id[qa_id]["source_or_generated"] == "official" for qa_id in row_roles if qa_id in provenance_by_id),
        "forget_role_rows": sum("forget" in roles for roles in row_roles.values()),
        "protected_eval_role_rows": sum("protected_eval" in roles for roles in row_roles.values()),
        "overlap_role_rows": sum(roles == {"forget", "protected_eval"} for roles in row_roles.values()),
        "missing_view_rows": missing_view_rows,
        "missing_request_refs": missing_requests,
        "view_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/api/eval_extension/eval_a_candidate_view/candidate_outputs_4000_eval_a.jsonl",
        "review_decisions_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/audit/bundle_eval_row_review_2026-07-18/review_decisions.jsonl",
        "gate_dependency": "formal calibration gate and actual human semantic decisions are required before changing review_status from pending_human_review",
    }
    (output_root / "review_inventory_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
