# Atomic-TOFU v2-pilot dataset construction completion report

Date: 2026-07-19

Status: `pilot_ready_with_known_evaluator_source_shift`

This report covers only the v2-pilot candidate revision and its derived audits, manifests, Eval-A view, and formal bundle hash inventory. It does not authorize or record v3 generation, training, or a formal Eval-A experiment.

## Scope and execution note

The unique project directory was `/home/zkzhang/unlearn/open-unlearning`. The requested `ssh hpc` connection could not be established in this execution environment: the alias failed DNS resolution, and the available login hostname failed public-key authentication. Work was therefore performed in the same local project directory. No source or project baseline was changed to compensate for this environment difference.

The pre-existing dirty worktree was recorded and left untouched. No `git clean`, `git reset`, `git checkout`, staging, commit, or cleanup was performed.

All artifacts in this report are candidate artifacts:

```text
candidate_only=true
model_lock=floating_alias
api_called=false
```

## Pipeline A and source coordination

Pipeline A, annotation, human-review packets, and request graph scopes were reused unchanged. The current coordination report is:

`data/atomic_tofu/v2.0-rc1/audit/source_pipeline_audit_2026-07-18.json`

Its status remains `passed_with_source_process_warning`. The current source parses as 4000 rows / 200 authors / 20 QAs per author, with contiguous author blocks and aligned `source_index`/`qa_id`; current author identity agrees across source, annotation, adjudicated annotation, review packets, and 1174 request scopes.

The current source SHA is `94fc1caa196658c1abbb586e61c25f4d09544be3a1943997e6e3c4f036a821ff`. The historical source report SHA is `667baef2f26f781f1328701d8ea54bf25123ff22fb2d98574c96cd991cb54086`; the mismatch is retained as a process warning. `source/tofu_full.jsonl` was not modified.

## v2-pilot 3200 revision

Parent revision:

- Frozen baseline: `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl`, SHA `7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`.
- Shape-repaired parent: `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/shape_repaired_53_2026-07-18/candidate_outputs_3200_shape_repaired_v1.jsonl`, SHA `114728c0a2dce0d1d29be0ee892dd512b2861fa7d3befc5f27dbdce215063f39`.

New revision:

- Candidate: `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/candidate_outputs_3200_v2_pilot_r1.jsonl`
- SHA256: `0d3ced300229ee611a7387c5e46bdc71db5d5c0e3ffdf46f6ef958c77f5e0b22`
- Revision manifest: `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_revision_manifest.json`
- Field provenance: `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_field_provenance.jsonl`
- Semantic/mechanical audit: `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_semantic_mechanical_audit_2026-07-19.json`
- Freeze record: `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_freeze_record_2026-07-19.json`

Exactly 8 IDs and 12 perturbation fields were revised. The 13 findings from the parent semantic audit are all explicitly closed. The other 53-row shape repair remains preserved; the frozen baseline and shape-repaired parent retain their original SHAs.

Candidate-level checks passed for 3200 rows, 3200 unique IDs, input-unit set/order, source question/answer payload alignment, five non-empty unique perturbations per row, authorized-field-only changes, provenance/hash correspondence, source-slot preservation, and target type continuity. Under the audit contract, correct-answer leakage, internal contradiction, source-slot omission, naturalness/punctuation failures in the 12 fields, non-bare-source/bare-value failures, and answer-shape mismatch are all zero. This is not a new human review and is not a semantic-zero-error claim.

The broad length gate remains failed and was not repaired mechanically. For all 16000 perturbations, ratio median/P10/P90 are `0.7714285714 / 0.5142857143 / 1.0`; 7121 fields are below 0.75 and 4453 below 0.65. For the 3060 formal-bundle generated rows / 15300 perturbations, ratio median/P10/P90 are `0.7727272727 / 0.5151515152 / 1.0`; 6782 fields are below 0.75 and 4239 below 0.65. The complete diagnostic is `data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_length_shape_diagnostic.json`.

## 4000-row Eval-A candidate view

A new versioned view was created without overwriting the older view:

- View: `data/atomic_tofu/v2.0-rc1/api/eval_extension/eval_a_candidate_view_v2_pilot_2026-07-19/candidate_outputs_4000_eval_a_v2_pilot.jsonl`
- SHA256: `3a9ba2c0ba1a4a0dbda94839117f962d5208538c87266caed36e18ecf31eb607`
- Field-level provenance: `data/atomic_tofu/v2.0-rc1/api/eval_extension/eval_a_candidate_view_v2_pilot_2026-07-19/candidate_outputs_4000_eval_a_v2_pilot.provenance.jsonl`
- Provenance SHA256: `0cff2722f23b51021b0fc2f261abef4bf2e5cdd3dd80a123b0558ba1cec44b57`
- Manifest: `data/atomic_tofu/v2.0-rc1/api/eval_extension/eval_a_candidate_view_v2_pilot_2026-07-19/eval_a_v2_pilot_view_manifest.json`

The view contains 4000 source-ordered rows: 3200 v2-pilot generated candidate rows and 800 official auxiliary-field rows. Official fields were not overwritten. Each row has field-level provenance and a persisted view-row hash.

## Formal bundle hash inventory

The new inventory is under `data/atomic_tofu/v2.0-rc1/audit/bundle_eval_row_review_v2_pilot_2026-07-19/`.

It records 39 bundles (27 pure, 12 balanced), 3420 unique Eval-A rows, 3060 generated rows, and 360 official rows. The existing bundle/request/forget/protected selection was read-only compared and preserved. New review decisions bind to the new view row hash and candidate hash.

The review coverage basis remains `user_manual_review_attestation`; the inventory does not claim 3420/3420 individually reviewed, and prior pending decisions/reviewer fields were preserved. The bundle content inventory SHA is `779e7b1bccfc651cb7973a8ac280b0c8345a6ac9ffb79532ee05218d1989ed03`; the new review decision inventory SHA is `96643d59f52250e45d6dd6dadaf50e93db7251678ca02180447378aee9d5d00f`.

## Hidden-anchor paired evaluator-source shift

The existing non-training paired run `89472` was used as behavioral evidence only: `results/eval/atomic_tofu_eval_a_anchor20_paired_20260719/89472/`.

It used the same full checkpoint/tokenizer/configuration for official and regenerated fields and evaluated 20 hidden anchors. The summary reports:

- paraphrased-answer avg-loss mean difference (generated - official): `-0.6384765625`;
- mean perturbed-answer avg-loss mean difference: `+0.3421484375`;
- truth-ratio mean difference: `-0.3987381408`;
- truth-ratio direction reversals: `3/20`;
- NaN/Inf failures: `0`;
- decision: `not_automatically_declared`.

This is evidence of evaluator-source shift, not behavioral equivalence. It is not a calibration-pass declaration and does not turn generated fields into official or gold fields.

## Release boundary

Eval-B has been removed from the mandatory Atomic-TOFU / BRIDGE-R v10 scope. It is optional future/appendix work and does not block this v2-pilot candidate artifact. The current state is pilot-ready with the known evaluator-source shift, not v3 final-ready. No v3 generation, training, Slurm submission, or formal Eval-A experiment was started in this step.
