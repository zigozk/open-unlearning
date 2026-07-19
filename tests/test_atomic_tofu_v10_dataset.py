import tempfile
import unittest
from pathlib import Path

from atomic_tofu.io import write_json, write_jsonl
from data.atomic_tofu_v10 import AtomicTOFUUnlearnDataset


class AtomicTOFUDatasetTests(unittest.TestCase):
    def test_active_pool_remains_the_request_scoped_same_author_complement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = [
                {
                    "qa_id": f"q{index}",
                    "author_id": "author_a",
                    "source_index": index,
                    "question": f"question {index}",
                    "answer": f"answer {index}",
                }
                for index in range(4)
            ]
            requests = [
                {
                    "request_id": "r0",
                    "author_id": "author_a",
                    "forget_qa_ids": ["q0"],
                    "protected_train_qa_ids": ["q1", "q2", "q3"],
                },
                {
                    "request_id": "r1",
                    "author_id": "author_a",
                    "forget_qa_ids": ["q1"],
                    "protected_train_qa_ids": ["q0", "q2", "q3"],
                },
            ]
            bundle = {"request_ids": ["r0", "r1"], "forget_qa_ids": ["q0", "q1"]}
            source_path = root / "source.jsonl"
            requests_path = root / "requests.jsonl"
            bundle_path = root / "bundle.json"
            write_jsonl(source_path, source)
            write_jsonl(requests_path, requests)
            write_json(bundle_path, bundle)

            dataset = AtomicTOFUUnlearnDataset(
                source_path=source_path,
                bundle_path=bundle_path,
                requests_path=requests_path,
                template_args=None,
                tokenizer=None,
                logical_k=8,
            )

            protected_ids, _ = dataset._protected_ids("q0", 0)
            self.assertEqual(set(dataset.retain_ids), {"q2", "q3"})
            self.assertEqual(set(dataset.active_protected_ids["r0"]), {"q1", "q2", "q3"})
            self.assertEqual(set(protected_ids), {"q1", "q2", "q3"})


if __name__ == "__main__":
    unittest.main()
