import unittest

from atomic_tofu.annotation import mock_annotation
from atomic_tofu.request_graph import compile_request


class RequestGraphTests(unittest.TestCase):
    def setUp(self):
        self.unit = {
            "payload": {
                "author_id": "tofu_author_000",
                "qas": [
                    {"qa_id": f"q{i}", "question": f"question {i}", "answer": f"answer {i}"}
                    for i in range(20)
                ],
            }
        }
        self.annotation = mock_annotation(self.unit)

    def test_single_and_multi_compile_complete_closures(self):
        author_qa_ids = {f"q{i}" for i in range(20)}
        single = compile_request(self.annotation, self.annotation["single_requests"][0], set(), author_qa_ids)
        multi = compile_request(self.annotation, self.annotation["multi_requests"][0], set(), author_qa_ids)
        self.assertTrue(single["request_id"].startswith("tofu_author_000::"))
        self.assertEqual(single["candidate_request_id"], self.annotation["single_requests"][0]["request_id_candidate"])
        self.assertEqual(single["cardinality_label"], "C1")
        self.assertEqual(multi["cardinality_label"], "C2")
        self.assertEqual(set(multi["forget_qa_ids"]), {"q0", "q1"})
        self.assertEqual(set(single["protected_train_qa_ids"]), author_qa_ids - {"q0"})
        self.assertEqual(set(multi["protected_train_qa_ids"]), author_qa_ids - {"q0", "q1"})
        self.assertEqual(single["protected_train_qa_ids"], single["protected_eval_qa_ids"])
        self.assertEqual(
            single["provenance"]["protected_pool_provenance"]["strategy"],
            "same_author_non_target_complement",
        )

    def test_reserved_overlap_marks_noncomparable_track(self):
        request = compile_request(
            self.annotation,
            self.annotation["single_requests"][0],
            {"q0"},
            {f"q{i}" for i in range(20)},
        )
        self.assertFalse(request["main_track_eligible"])


if __name__ == "__main__":
    unittest.main()
