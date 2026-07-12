import tempfile
import unittest
from pathlib import Path

from atomic_tofu.io import read_jsonl, sha256_json, write_jsonl
from atomic_tofu.pipeline import run_units


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


if __name__ == "__main__":
    unittest.main()
