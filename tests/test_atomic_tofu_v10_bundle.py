import unittest

from atomic_tofu.bundles import exact_union_bundle


class BundleTests(unittest.TestCase):
    def test_exact_union_never_truncates_closure(self):
        requests = [
            {"request_id": "r1", "target_atom_ids": ["a"], "forget_qa_ids": ["q1", "q2"]},
            {"request_id": "r2", "target_atom_ids": ["b"], "forget_qa_ids": ["q2", "q3"]},
            {"request_id": "r3", "target_atom_ids": ["c"], "forget_qa_ids": ["q4"]},
        ]
        selected = exact_union_bundle(requests, 3, seed=0, cardinalities={1})
        union = set().union(*(set(request["forget_qa_ids"]) for request in selected))
        self.assertEqual(len(union), 3)

    def test_impossible_exact_union_fails(self):
        requests = [{"request_id": "r", "target_atom_ids": ["a"], "forget_qa_ids": ["q1", "q2"]}]
        with self.assertRaises(ValueError):
            exact_union_bundle(requests, 1, seed=0)


if __name__ == "__main__":
    unittest.main()

