from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("data/atomic_tofu/v2.0-rc1")
VIEW_ROOT = ROOT / "api/eval_extension/eval_a_candidate_view_v2_pilot_2026-07-19"
VIEW = VIEW_ROOT / "candidate_outputs_4000_eval_a_v2_pilot.jsonl"
VIEW_PROVENANCE = VIEW_ROOT / "candidate_outputs_4000_eval_a_v2_pilot.provenance.jsonl"
OLD_REVIEW = ROOT / "audit/bundle_eval_row_review_2026-07-18/review_decisions.jsonl"
OUTPUT_ROOT = ROOT / "audit/bundle_eval_row_review_v2_pilot_2026-07-19"
REVIEW_OUTPUT = OUTPUT_ROOT / "review_decisions_v2_pilot.jsonl"
BUNDLE_OUTPUT = OUTPUT_ROOT / "bundle_content_hash_inventory.jsonl"
MANIFEST = OUTPUT_ROOT / "bundle_hash_inventory_manifest.json"
REVIEW_SHA = OUTPUT_ROOT / "review_decisions_v2_pilot.jsonl.sha256"
BUNDLE_SHA = OUTPUT_ROOT / "bundle_content_hash_inventory.jsonl.sha256"

EXPECTED_VIEW_SHA = "3a9ba2c0ba1a4a0dbda94839117f962d5208538c87266caed36e18ecf31eb607"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    for path in (VIEW, VIEW_PROVENANCE, OLD_REVIEW):
        if not path.exists():
            raise FileNotFoundError(path)
    if file_hash(VIEW) != EXPECTED_VIEW_SHA:
        raise RuntimeError("v2-pilot Eval-A view SHA changed")

    view_rows = read_jsonl(VIEW)
    provenance_rows = read_jsonl(VIEW_PROVENANCE)
    old_review_rows = read_jsonl(OLD_REVIEW)
    view_by_id = {row["qa_id"]: row for row in view_rows}
    provenance_by_id = {row["qa_id"]: row for row in provenance_rows}
    old_by_id = {row["qa_id"]: row for row in old_review_rows}
    if len(view_rows) != 4000 or len(provenance_rows) != 4000:
        raise RuntimeError("v2-pilot view/provenance must contain 4000 rows")
    if len(old_review_rows) != 3420:
        raise RuntimeError("old formal review inventory must contain 3420 rows")

    request_path = ROOT / "request_graph/requests.jsonl"
    request_by_id = {row["request_id"]: row for row in read_jsonl(request_path)}
    row_roles: dict[str, set[str]] = defaultdict(set)
    row_bundles: dict[str, set[str]] = defaultdict(set)
    bundle_records: list[dict[str, Any]] = []
    missing_requests: list[dict[str, str]] = []
    bundle_paths = sorted(
        path for family in ("pure", "balanced")
        for path in (ROOT / "bundles" / family).rglob("*.json")
        if not path.name.endswith(".manifest.json")
    )
    for path in bundle_paths:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        bundle_id = bundle["bundle_id"]
        request_ids = list(bundle.get("request_ids", []))
        forget_ids = list(bundle.get("forget_qa_ids", []))
        protected_ids: list[str] = []
        for qa_id in forget_ids:
            row_roles[qa_id].add("forget")
            row_bundles[qa_id].add(bundle_id)
        for request_id in request_ids:
            request = request_by_id.get(request_id)
            if request is None:
                missing_requests.append({"bundle_id": bundle_id, "request_id": request_id})
                continue
            for qa_id in request.get("protected_eval_qa_ids", []):
                protected_ids.append(qa_id)
                row_roles[qa_id].add("protected_eval")
                row_bundles[qa_id].add(bundle_id)
        family = "balanced" if "/bundles/balanced/" in str(path) else "pure"
        bundle_records.append({
            "bundle_id": bundle_id,
            "family": family,
            "path": str(path),
            "bundle_sha256": file_hash(path),
            "request_ids": request_ids,
            "forget_qa_ids": forget_ids,
            "protected_eval_qa_ids": protected_ids,
            "content_preserved": True,
            "selection_source": "existing bundles read-only; no bundle/request/forget/protected selection changed",
        })

    if len(bundle_records) != 39:
        raise RuntimeError(f"expected 39 bundles, got {len(bundle_records)}")
    if missing_requests:
        raise RuntimeError(f"missing request refs: {missing_requests[:3]}")
    if len(row_roles) != 3420:
        raise RuntimeError(f"expected 3420 unique formal rows, got {len(row_roles)}")
    if set(row_roles) != set(old_by_id):
        raise RuntimeError("formal inventory row set changed")

    decisions: list[dict[str, Any]] = []
    comparison_failures: list[str] = []
    for qa_id in sorted(row_roles):
        old = old_by_id[qa_id]
        view = view_by_id.get(qa_id)
        provenance = provenance_by_id.get(qa_id)
        if view is None or provenance is None:
            comparison_failures.append(f"missing view/provenance: {qa_id}")
            continue
        if sorted(old["bundle_ids"]) != sorted(row_bundles[qa_id]) or old["bundle_roles"] != sorted(row_roles[qa_id]):
            comparison_failures.append(f"bundle role/selection changed: {qa_id}")
        if old["source_or_generated"] != provenance["source_or_generated"]:
            comparison_failures.append(f"source/generated classification changed: {qa_id}")
        candidate_hash = canonical_hash(view)
        decisions.append({
            "qa_id": qa_id,
            "bundle_roles": sorted(row_roles[qa_id]),
            "bundle_ids": sorted(row_bundles[qa_id]),
            "source_or_generated": provenance["source_or_generated"],
            "candidate_hash": candidate_hash,
            "candidate_hash_algorithm": "sha256(canonical_eval_a_row_json_utf8_sort_keys)",
            "view_row_sha256": provenance["view_row_sha256"],
            "generated_candidate_sha256": provenance.get("generated_candidate_sha256"),
            "official_sidecar_sha256": provenance.get("official_sidecar_sha256"),
            "candidate_revision": provenance.get("candidate_revision"),
            "review_status": old["review_status"],
            "decision": old["decision"],
            "reviewer": old["reviewer"],
            "review_coverage_basis": "user_manual_review_attestation",
            "individual_row_review_claim": False,
            "review_decision_binding": {
                "candidate_hash": candidate_hash,
                "view_row_sha256": provenance["view_row_sha256"],
                "candidate_only": True,
                "model_lock": "floating_alias",
            },
            "mechanical_context": {
                "official_fields_overwritten": False,
                "human_review_status_from_source": old.get("mechanical_context", {}).get("human_review_status_from_source"),
                "v2_pilot_view_sha256": EXPECTED_VIEW_SHA,
            },
        })
    if comparison_failures:
        raise RuntimeError(f"review inventory comparison failures: {comparison_failures[:10]}")
    if any(row["candidate_hash"] != row["view_row_sha256"] for row in decisions):
        raise RuntimeError("candidate hash/view row hash mismatch")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(REVIEW_OUTPUT, decisions)
    write_jsonl(BUNDLE_OUTPUT, bundle_records)
    review_sha = file_hash(REVIEW_OUTPUT)
    bundle_sha = file_hash(BUNDLE_OUTPUT)
    REVIEW_SHA.write_text(f"{review_sha}  {REVIEW_OUTPUT.name}\n", encoding="utf-8")
    BUNDLE_SHA.write_text(f"{bundle_sha}  {BUNDLE_OUTPUT.name}\n", encoding="utf-8")
    manifest = {
        "artifact": "Atomic-TOFU v2-pilot formal bundle Eval-A row hash inventory",
        "status": "inventory_rebuilt_hash_bound_pending_human_review",
        "revision_date": "2026-07-19",
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called": False,
        "view_path": str(VIEW),
        "view_sha256": EXPECTED_VIEW_SHA,
        "review_decisions_path": str(REVIEW_OUTPUT),
        "review_decisions_sha256": review_sha,
        "bundle_content_inventory_path": str(BUNDLE_OUTPUT),
        "bundle_content_inventory_sha256": bundle_sha,
        "bundle_count": len(bundle_records),
        "bundle_count_by_family": {
            "pure": sum(item["family"] == "pure" for item in bundle_records),
            "balanced": sum(item["family"] == "balanced" for item in bundle_records),
        },
        "unique_eval_rows": len(decisions),
        "persisted_review_decisions": len(decisions),
        "generated_rows": sum(item["source_or_generated"] == "generated" for item in decisions),
        "official_rows": sum(item["source_or_generated"] == "official" for item in decisions),
        "forget_role_rows": sum("forget" in row_roles[qa_id] for qa_id in row_roles),
        "protected_eval_role_rows": sum("protected_eval" in row_roles[qa_id] for qa_id in row_roles),
        "overlap_role_rows": sum(row_roles[qa_id] == {"forget", "protected_eval"} for qa_id in row_roles),
        "bundle_selection_unchanged_from_existing_inventory": not comparison_failures,
        "bundle_content_read_only_hashes_recorded": True,
        "review_coverage_basis": "user_manual_review_attestation",
        "individual_bundle_row_review_claim": False,
        "human_review_completed": False,
        "decision_status_preserved_from_prior_inventory": True,
        "official_fields_overwritten": False,
        "eval_b_scope": "removed_from_mandatory_v2_pilot_release_scope",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
