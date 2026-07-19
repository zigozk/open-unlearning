from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json


BALANCED_SPEC_ID = "balanced_v1"
SEEDS = (0, 1, 2)
FAMILY_CELLS = {
    "Single": ("C1-E-Low", "C1-E-Medium", "C1-E-High"),
    "Multi": ("C2-E-Low", "C2-E-Medium", "C2-E-High", "C3-E-Low", "C3-E-Medium", "C3-E-High"),
}
QUOTAS = {
    "Single": {
        40: ((14, 13, 13), (13, 14, 13), (13, 13, 14)),
        200: ((67, 67, 66), (66, 67, 67), (67, 66, 67)),
    },
    "Multi": {
        40: ((7, 7, 6, 7, 6, 7), (6, 7, 7, 7, 7, 6), (7, 6, 7, 6, 7, 7)),
        200: ((34, 33, 33, 33, 34, 33), (33, 34, 33, 33, 33, 34), (33, 33, 34, 34, 33, 33)),
    },
}


def balanced_spec() -> dict[str, Any]:
    return {
        "spec_id": BALANCED_SPEC_ID,
        "quota_unit": "unique_forget_qa_owner_attribution",
        "families": {family: list(cells) for family, cells in FAMILY_CELLS.items()},
        "quota_tables": {
            family: {str(size): [list(row) for row in table] for size, table in sizes.items()}
            for family, sizes in QUOTAS.items()
        },
        "attribution": {
            "owner_rule": "argmin sha256(bundle_seed || qa_id || request_id)",
            "encoding": "f'{bundle_seed}||{qa_id}||{request_id}' UTF-8",
            "scope": "quota and reporting only; does not modify forget closure or protected pools",
        },
        "constraints": {
            # A 7-QA C2-E-High closure exists for one author; a strict 10%
            # cap would make the prescribed 40-QA seed quota infeasible.  The
            # 40-QA cap is therefore pre-registered at 8, while the 200-QA
            # cap remains 10% (20 QA).
            "max_per_author_qa": {"40": 8, "200": 20},
            "min_relation_categories": {"40": 3, "200": 4},
            "max_pairwise_closure_overlap_ratio": 0.25,
            "required_distinct_seeds": 3,
            "no_reserved_anchor_overlap": True,
            "complete_request_closures": True,
            "selection_uses_model_results": False,
        },
        "pure_200_unavailable": {
            "C1-E-High": {"status": "unavailable_at_200", "reason": "eligible complete closure union < 200"},
            "C2-E-Low": {"status": "unavailable_at_200", "reason": "eligible complete closure union < 200"},
            "C3-E-High": {"status": "unavailable_at_200", "reason": "eligible complete closure union < 200"},
            "C3-E-Low": {"status": "unavailable_at_200", "reason": "eligible complete closure union < 200"},
        },
    }


def _cell(request: dict[str, Any]) -> str:
    return f"C{len(request.get('target_atom_ids', []))}-{request.get('entanglement_level')}"


def _category(relation: str) -> str:
    value = relation.lower()
    if any(token in value for token in ("birth", "place", "upbringing", "grew_up")):
        return "birth"
    if any(token in value for token in ("father", "mother", "parent", "family")):
        return "family"
    if any(token in value for token in ("book", "work", "title", "publication", "novel", "authored", "translation")):
        return "works"
    if any(token in value for token in ("award", "prize", "acclaim", "reception", "recognition")):
        return "recognition"
    if any(token in value for token in ("education", "degree", "school", "university")):
        return "education"
    return "other"


def _request_hash_key(bundle_seed: int, qa_id: str, request_id: str) -> str:
    payload = f"{bundle_seed}||{qa_id}||{request_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _owner_attribution(selected: list[dict[str, Any]], seed: int) -> dict[str, dict[str, str]]:
    by_qa: dict[str, list[dict[str, Any]]] = {}
    for request in selected:
        for qa_id in request["forget_qa_ids"]:
            by_qa.setdefault(qa_id, []).append(request)
    owners: dict[str, dict[str, str]] = {}
    for qa_id, requests in by_qa.items():
        owner = min(requests, key=lambda row: _request_hash_key(seed, qa_id, row["request_id"]))
        owners[qa_id] = {
            "owner_request_id": owner["request_id"],
            "owner_author_id": owner["author_id"],
            "owner_cell": _cell(owner),
            "owner_hash": _request_hash_key(seed, qa_id, owner["request_id"]),
        }
    return dict(sorted(owners.items()))


def _random_exact_cell_solutions(
    candidates: list[dict[str, Any]],
    target: int,
    seed: int,
    *,
    blocked_qa_ids: set[str] | None = None,
    max_solutions: int = 120,
    max_restarts: int = 6000,
) -> list[tuple[list[dict[str, Any]], set[str]]]:
    blocked_qa_ids = blocked_qa_ids or set()
    candidates = [row for row in candidates if not (set(row["forget_qa_ids"]) & blocked_qa_ids)]
    if len(set().union(*(set(row["forget_qa_ids"]) for row in candidates))) < target:
        return []
    # Sparse cells are small enough for an exact subset search.  This also
    # catches a single request whose complete closure is exactly the quota
    # (important for the C2-E-High 40-QA quota).
    exact: list[tuple[list[dict[str, Any]], set[str]]] = []
    exact_signatures: set[tuple[str, ...]] = set()
    ordered = sorted(candidates, key=lambda row: (len(row["forget_qa_ids"]), row["request_id"]))

    def dfs(index: int, selected: list[dict[str, Any]], union: set[str]) -> None:
        if len(exact) >= max_solutions or index >= len(ordered):
            return
        if len(union) == target:
            signature = tuple(sorted(row["request_id"] for row in selected))
            if signature not in exact_signatures:
                exact_signatures.add(signature)
                exact.append((list(selected), set(union)))
            return
        for position in range(index, len(ordered)):
            request = ordered[position]
            next_union = union | set(request["forget_qa_ids"])
            if len(next_union) > target:
                continue
            dfs(position + 1, selected + [request], next_union)
            if len(exact) >= max_solutions:
                return

    if len(candidates) <= 25 or target <= 10:
        dfs(0, [], set())
        if exact:
            return exact
    rng = random.Random(seed)
    solutions: list[tuple[list[dict[str, Any]], set[str]]] = list(exact)
    signatures: set[tuple[str, ...]] = set(exact_signatures)
    for _ in range(max_restarts):
        order = candidates[:]
        rng.shuffle(order)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        union: set[str] = set()
        while len(union) < target:
            remaining = target - len(union)
            options = []
            for request in order:
                if request["request_id"] in selected_ids:
                    continue
                gain = len(set(request["forget_qa_ids"]) - union)
                if 0 < gain <= remaining:
                    options.append((request, gain))
            if not options:
                break
            options.sort(key=lambda item: item[1])
            # Retain small and exact-residual gains; taking only max-gain
            # requests cannot discover 3+4=7 solutions in sparse cells.
            window = options[: min(len(options), max(20, len(options) // 3))]
            request, _ = rng.choice(window)
            selected.append(request)
            selected_ids.add(request["request_id"])
            union.update(request["forget_qa_ids"])
        if len(union) != target:
            continue
        signature = tuple(sorted(selected_ids))
        if signature in signatures:
            continue
        signatures.add(signature)
        solutions.append((selected, union))
        if len(solutions) >= max_solutions:
            break
    return solutions


def _find_bundle(
    candidates_by_cell: dict[str, list[dict[str, Any]]],
    cells: tuple[str, ...],
    quotas: tuple[int, ...],
    seed: int,
    *,
    excluded_signatures: set[tuple[str, ...]],
) -> list[dict[str, Any]]:
    # Build independent exact cell pools and combine disjointly.  This makes
    # the owner attribution equal to the requested quota while retaining the
    # hash rule for any within-cell overlap.
    pools: dict[str, list[tuple[list[dict[str, Any]], set[str], Counter[str]]]] = {}
    for index, (cell, quota) in enumerate(zip(cells, quotas)):
        raw_solutions = _random_exact_cell_solutions(
            candidates_by_cell[cell], quota, seed * 1009 + index,
            max_solutions=160 if quota <= 40 else 50,
            max_restarts=10000 if quota <= 40 else 3500,
        )
        max_allowed = 8 if sum(quotas) <= 40 else 20
        pools[cell] = []
        for requests, union in raw_solutions:
            owners = _owner_attribution(requests, seed)
            author_counts = Counter(owner["owner_author_id"] for owner in owners.values())
            if max(author_counts.values(), default=0) <= max_allowed:
                pools[cell].append((requests, union, author_counts))
        if not pools[cell]:
            raise ValueError(f"no exact cell solution for {cell} quota={quota}")
    # Rare cells first reduces backtracking for C3-E-High/C3-E-Low.
    order = sorted(cells, key=lambda cell: (len(pools[cell]), cell))
    rng = random.Random(seed * 9176 + 19)
    for cell in order:
        rng.shuffle(pools[cell])

    max_allowed = 8 if sum(quotas) <= 40 else 20

    def search(
        index: int,
        used: set[str],
        selected: list[dict[str, Any]],
        author_counts: Counter[str],
    ) -> list[dict[str, Any]] | None:
        if index == len(order):
            signature = tuple(sorted(request["request_id"] for request in selected))
            return None if signature in excluded_signatures else selected
        for requests, union, cell_author_counts in pools[order[index]]:
            if union & used:
                continue
            updated_authors = author_counts + cell_author_counts
            if max(updated_authors.values(), default=0) > max_allowed:
                continue
            result = search(index + 1, used | union, selected + requests, updated_authors)
            if result is not None:
                return result
        return None

    result = search(0, set(), [], Counter())
    if result is None:
        raise ValueError(f"no disjoint exact balanced solution for seed={seed}")
    return result


def _validate_bundle(
    selected: list[dict[str, Any]],
    *,
    family: str,
    size: int,
    seed: int,
    cells: tuple[str, ...],
    quotas: tuple[int, ...],
    source_ids: set[str],
    reserved_ids: set[str],
    spec: dict[str, Any],
) -> dict[str, Any]:
    forget = set().union(*(set(row["forget_qa_ids"]) for row in selected))
    if len(forget) != size:
        raise ValueError(f"exact union is {len(forget)}, expected {size}")
    if any(set(row["forget_qa_ids"]) & reserved_ids for row in selected):
        raise ValueError("reserved official anchor overlap")
    if any(not set(row["forget_qa_ids"]) <= source_ids for row in selected):
        raise ValueError("unknown forget QA")
    owners = _owner_attribution(selected, seed)
    contributions = Counter(owner["owner_cell"] for owner in owners.values())
    expected = dict(zip(cells, quotas))
    if dict(contributions) != expected:
        raise ValueError(f"owner quota mismatch: got {dict(contributions)}, expected {expected}")
    author_contributions = Counter(owner["owner_author_id"] for owner in owners.values())
    max_author = max(author_contributions.values(), default=0)
    max_allowed = spec["constraints"]["max_per_author_qa"][str(size)]
    if max_author > max_allowed:
        raise ValueError(f"author contribution {max_author} exceeds {max_allowed}")
    categories = Counter(
        _category(relation)
        for row in selected
        for relation in row.get("target_atom_relations", {}).values()
    )
    min_categories = spec["constraints"]["min_relation_categories"][str(size)]
    if len(categories) < min_categories:
        raise ValueError(f"only {len(categories)} relation categories, need {min_categories}")
    closure_sum = sum(len(set(row["forget_qa_ids"])) for row in selected)
    pairwise_overlap = sum(
        len(set(left["forget_qa_ids"]) & set(right["forget_qa_ids"]))
        for index, left in enumerate(selected)
        for right in selected[index + 1:]
    )
    overlap_ratio = pairwise_overlap / closure_sum if closure_sum else 0.0
    if overlap_ratio > spec["constraints"]["max_pairwise_closure_overlap_ratio"]:
        raise ValueError(f"pairwise closure overlap ratio {overlap_ratio:.4f} exceeds bound")
    owner_rows = [
        {"qa_id": qa_id, **owner}
        for qa_id, owner in owners.items()
    ]
    owner_hash = sha256_json(owner_rows)
    return {
        "bundle_id": f"balanced_{family}_{size}_seed_{seed}",
        "family": family,
        "seed": seed,
        "target_qa_count": size,
        "request_ids": [row["request_id"] for row in selected],
        "forget_qa_ids": sorted(forget),
        "request_count": len(selected),
        "author_ids": sorted({row["author_id"] for row in selected}),
        "cell_qa_contribution": dict(sorted(contributions.items())),
        "cell_request_counts": dict(sorted(Counter(_cell(row) for row in selected).items())),
        "author_qa_contribution": dict(sorted(author_contributions.items())),
        "relation_category_counts": dict(sorted(categories.items())),
        "closure_qa_sum": closure_sum,
        "closure_overlap_qa_count": pairwise_overlap,
        "closure_overlap_ratio": overlap_ratio,
        "owner_attribution": owner_rows,
        "owner_attribution_sha256": owner_hash,
        "constraints": {
            "exact_union": len(forget) == size,
            "complete_request_closures": True,
            "no_reserved_anchor_overlap": True,
            "max_per_author_qa": max_allowed,
            "min_relation_categories": min_categories,
            "max_pairwise_closure_overlap_ratio": spec["constraints"]["max_pairwise_closure_overlap_ratio"],
            "owner_quota": expected,
        },
        "selection_uses_model_results": False,
    }


def compile_balanced_bundles(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    spec = balanced_spec()
    spec_hash = sha256_json(spec)
    write_json(root / "bundle_specs" / f"{BALANCED_SPEC_ID}.json", {**spec, "spec_sha256": spec_hash})
    requests = read_jsonl(root / "request_graph" / "requests.jsonl")
    eligible = [row for row in requests if row.get("main_track_eligible") is True]
    source_ids = {row["qa_id"] for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    reserved_ids = set(read_json(root / "official_anchors" / "reserved_utility_anchors.json")["qa_ids"])
    report: dict[str, Any] = {
        "spec_id": BALANCED_SPEC_ID,
        "spec_sha256": spec_hash,
        "request_graph_sha256": sha256_json([
            {key: row[key] for key in ("request_id", "target_atom_ids", "forget_qa_ids")}
            for row in sorted(eligible, key=lambda value: value["request_id"])
        ]),
        "request_count_all": len(requests),
        "request_count_main_track_eligible": len(eligible),
        "request_count_ineligible": len(requests) - len(eligible),
        "pure_200_unavailable": spec["pure_200_unavailable"],
        "families": {},
        "status": "completed",
    }
    for family, cells in FAMILY_CELLS.items():
        candidates_by_cell = {cell: [row for row in eligible if _cell(row) == cell] for cell in cells}
        family_report: dict[str, Any] = {}
        for size in (40, 200):
            size_report = {"target_qa_count": size, "seeds": [], "status": "completed"}
            signatures: set[tuple[str, ...]] = set()
            for seed, quota_row in enumerate(QUOTAS[family][size]):
                try:
                    selected = _find_bundle(
                        candidates_by_cell, cells, quota_row, seed,
                        excluded_signatures=signatures,
                    )
                    bundle = _validate_bundle(
                        selected, family=family, size=size, seed=seed, cells=cells,
                        quotas=quota_row, source_ids=source_ids, reserved_ids=reserved_ids, spec=spec,
                    )
                except ValueError as error:
                    size_report["status"] = "failed"
                    size_report["seeds"].append({"seed": seed, "status": "failed", "error": str(error)})
                    continue
                signature = tuple(sorted(bundle["request_ids"]))
                signatures.add(signature)
                bundle["spec_id"] = BALANCED_SPEC_ID
                bundle["spec_sha256"] = spec_hash
                bundle["request_graph_sha256"] = report["request_graph_sha256"]
                bundle_path = root / "bundles" / "balanced" / family / str(size) / f"seed_{seed}.json"
                manifest_path = bundle_path.with_suffix(".manifest.json")
                write_json(bundle_path, {key: value for key, value in bundle.items() if key != "owner_attribution"})
                write_json(manifest_path, bundle)
                size_report["seeds"].append({
                    "seed": seed,
                    "status": "feasible",
                    "bundle_path": str(bundle_path.relative_to(root)),
                    "manifest_path": str(manifest_path.relative_to(root)),
                    "forget_qa_count": len(bundle["forget_qa_ids"]),
                    "owner_attribution_sha256": bundle["owner_attribution_sha256"],
                })
            if len(signatures) != len(SEEDS):
                size_report["status"] = "failed"
                size_report["distinct_solution_count"] = len(signatures)
            family_report[str(size)] = size_report
        report["families"][family] = family_report
    if any(size_report["status"] != "completed" for family in report["families"].values() for size_report in family.values()):
        report["status"] = "failed"
    write_json(root / "audit" / "balanced_bundle_feasibility_report.json", report)
    return report
