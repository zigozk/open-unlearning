import tempfile
import unittest
from pathlib import Path

from atomic_tofu.annotation import annotation_schema
from atomic_tofu import SCHEMA_VERSION
from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_jsonl
from atomic_tofu.pipeline import run_units
from atomic_tofu.reporting import build_annotation_review_report


def annotation_unit(author_id: str) -> dict:
    payload = {
        "author_id": author_id,
        "qas": [
            {"qa_id": f"{author_id}_{index}", "question": f"Q{index}", "answer": f"A{index}"}
            for index in range(3)
        ],
    }
    return {"unit_id": author_id, "content_sha256": sha256_json(payload), "payload": payload}


class PipelineSubsetTests(unittest.TestCase):
    def test_disjoint_resume_subsets_preserve_prior_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "api" / "annotation" / "input_units.jsonl",
                [annotation_unit("author_a"), annotation_unit("author_b")],
            )
            first = run_units(root, "annotation", "mock", resume=True, unit_ids={"author_a"})
            second = run_units(root, "annotation", "mock", resume=True, unit_ids={"author_b"})

            self.assertEqual(first["total_completed"], 1)
            self.assertEqual(second["completed"], 1)
            self.assertEqual(second["total_completed"], 2)
            self.assertEqual(
                {row["unit_id"] for row in read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")},
                {"author_a", "author_b"},
            )

    def test_new_model_subset_preserves_prior_model_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "api" / "annotation" / "input_units.jsonl",
                [annotation_unit("author_a"), annotation_unit("author_b")],
            )
            run_units(root, "annotation", "mock", resume=True, unit_ids={"author_a"})
            previous = read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")[0]
            previous["generation_sha256"] = "different-model-generation"
            write_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl", [previous])

            run_units(root, "annotation", "mock", resume=True, unit_ids={"author_b"})

            outputs = {row["unit_id"]: row for row in read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")}
            self.assertEqual(set(outputs), {"author_a", "author_b"})
            self.assertEqual(outputs["author_a"]["generation_sha256"], "different-model-generation")

    def test_legacy_prompt_cache_is_archived_and_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = annotation_unit("author_a")
            write_jsonl(root / "api" / "annotation" / "input_units.jsonl", [unit])
            legacy = {
                "record_id": "annotation_author_a_legacy",
                "unit_id": "author_a",
                "input_sha256": unit["content_sha256"],
                "schema_sha256": sha256_json(annotation_schema()),
                "candidate": {"legacy": True},
                "provenance": {"schema_version": SCHEMA_VERSION},
            }
            write_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl", [legacy])

            report = run_units(root, "annotation", "mock", resume=True, unit_ids={"author_a"})

            current = read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")
            self.assertEqual(report["completed"], 1)
            self.assertEqual(len(current), 1)
            self.assertIn("generation_sha256", current[0])
            self.assertFalse(current[0]["candidate"].get("legacy", False))
            self.assertEqual(
                read_json(root / "api" / "annotation" / "attempts" / "annotation_author_a_legacy.json"),
                legacy,
            )

    def test_review_report_contains_atoms_requests_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = annotation_unit("author_a")
            source = [
                {"author_id": "author_a", **qa}
                for qa in unit["payload"]["qas"]
            ]
            write_jsonl(root / "source" / "tofu_full.jsonl", source)
            write_jsonl(root / "api" / "annotation" / "input_units.jsonl", [unit])
            run_units(root, "annotation", "mock", resume=True, unit_ids={"author_a"})

            report_path = build_annotation_review_report(root, "author_a")

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("## Original unchanged QAs", report)
            self.assertIn("## Candidate atoms", report)
            self.assertIn("## Candidate Single requests", report)
            self.assertIn("## Candidate Multi requests", report)
            self.assertIn("author_a_candidate_atom_00", report)


if __name__ == "__main__":
    unittest.main()
