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
        single = compile_request(self.annotation, self.annotation["single_requests"][0], set())
        multi = compile_request(self.annotation, self.annotation["multi_requests"][0], set())
        self.assertEqual(single["cardinality_label"], "C1")
        self.assertEqual(multi["cardinality_label"], "C2")
        self.assertEqual(set(multi["forget_qa_ids"]), {"q0", "q1"})
        self.assertNotIn("q0", multi["protected_train_qa_ids"])
        self.assertNotIn("q1", multi["protected_train_qa_ids"])

    def test_reserved_overlap_marks_noncomparable_track(self):
        request = compile_request(self.annotation, self.annotation["single_requests"][0], {"q0"})
        self.assertFalse(request["main_track_eligible"])


if __name__ == "__main__":
    unittest.main()

