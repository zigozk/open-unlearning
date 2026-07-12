import unittest

from atomic_tofu.controls import compile_matched_entity_control, compile_random_qa_control


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.source = [
            {
                "qa_id": f"q{author:03d}_{offset:02d}",
                "author_id": f"a{author:03d}",
                "question": f"question {author} {offset}",
                "answer": f"answer {author} {offset}",
            }
            for author in range(12)
            for offset in range(20)
        ]

    def test_matched_entity_exact_size(self):
        result = compile_matched_entity_control(self.source, [f"a{i:03d}" for i in range(12)], 40, 0)
        self.assertEqual(len(result["forget_qa_ids"]), 40)
        self.assertEqual(len(result["author_ids"]), 2)

    def test_random_control_matches_per_author_count(self):
        atomic = {"q000_00", "q000_01", "q001_00"}
        result = compile_random_qa_control(self.source, atomic, set(), 0, {row["qa_id"]: "mid" for row in self.source})
        self.assertEqual(result["target_qa_count"], 3)
        self.assertEqual(result["per_author_count"], {"a000": 2, "a001": 1})
        self.assertEqual(result["formal_status"], "ready")


if __name__ == "__main__":
    unittest.main()

