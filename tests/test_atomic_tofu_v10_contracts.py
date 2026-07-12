import unittest

from atomic_tofu.annotation import ANNOTATION_SYSTEM_PROMPT, mock_annotation
from atomic_tofu.contracts import cvar, score_entanglement, validate_request
from atomic_tofu.policies import apply_author_name_target_policy


class ContractTests(unittest.TestCase):
    def test_single_levels_use_structure_not_relation_name(self):
        low = {"target_atom_ids": ["a"], "entanglement_dimensions": {"exposure": 0, "dependency": 0, "co_carriage": 1, "protected_proximity": 1}}
        high = {"target_atom_ids": ["a"], "entanglement_dimensions": {"exposure": 2, "dependency": 2, "co_carriage": 1, "protected_proximity": 1}}
        self.assertEqual(score_entanglement(low)["entanglement_level"], "E-Low")
        self.assertEqual(score_entanglement(high)["entanglement_level"], "E-High")

    def test_multi_joint_shift_forces_high(self):
        request = {
            "target_atom_ids": ["a", "b"],
            "entanglement_dimensions": {
                "exposure": 0, "dependency": 0, "co_carriage": 0, "protected_proximity": 0,
                "closure_overlap": 0, "joint_role_shift": 2,
            },
            "max_constituent_single_score": 1,
        }
        self.assertEqual(score_entanglement(request)["entanglement_level"], "E-High")

    def test_reserved_anchor_and_protected_overlap_rejected(self):
        request = {
            "target_atom_ids": ["a"], "forget_qa_ids": ["q1"], "protected_train_qa_ids": ["q1"],
            "entanglement_dimensions": {"exposure": 0, "dependency": 0, "co_carriage": 0, "protected_proximity": 0},
        }
        errors = validate_request(request, {"q1", "q2"}, {"q1"})
        self.assertTrue(any("protected" in error for error in errors))
        self.assertTrue(any("anchors" in error for error in errors))

    def test_cvar20(self):
        self.assertEqual(cvar([1, 2, 3, 4, 5], 0.2), 5)

    def test_prompt_and_validator_require_closure_external_protected_qas(self):
        self.assertIn("outside the union of that request's per_atom_closures", ANNOTATION_SYSTEM_PROMPT)
        unit = {
            "payload": {
                "author_id": "author_a",
                "qas": [
                    {"qa_id": f"q{index}", "question": f"Q{index}", "answer": f"A{index}"}
                    for index in range(3)
                ],
            },
        }
        annotation = mock_annotation(unit)
        request = annotation["single_requests"][0]
        request["protected_train_qa_ids"] = [request["per_atom_closures"][0]["qa_ids"][0]]
        from atomic_tofu.contracts import validate_annotation
        errors = validate_annotation(annotation, {"q0", "q1", "q2"})
        self.assertTrue(any("protected QAs overlap closure" in error for error in errors))

    def test_author_name_target_request_is_excluded_without_changing_other_requests(self):
        annotation = {
            "atoms": [
                {"atom_id_candidate": "name", "relation": "full_name"},
                {"atom_id_candidate": "genre", "relation": "genre"},
            ],
            "single_requests": [
                {"request_id_candidate": "name_request", "target_atom_ids": ["name"]},
                {"request_id_candidate": "genre_request", "target_atom_ids": ["genre"]},
            ],
            "multi_requests": [{"request_id_candidate": "mixed_request", "target_atom_ids": ["name", "genre"]}],
        }
        effective, exclusions = apply_author_name_target_policy(annotation)
        self.assertEqual([request["request_id_candidate"] for request in effective["single_requests"]], ["genre_request"])
        self.assertEqual(effective["multi_requests"], [])
        self.assertEqual({item["request_id_candidate"] for item in exclusions}, {"name_request", "mixed_request"})

    def test_compact_schema_uses_four_digit_qa_ids_without_verbose_relation_fields(self):
        from atomic_tofu.annotation import annotation_schema
        relation = annotation_schema()["properties"]["atoms"]["items"]["properties"]["qa_relations"]["items"]
        self.assertEqual(relation["required"], ["qa_id", "role"])
        self.assertNotIn("reason", relation["properties"])
        self.assertEqual(relation["properties"]["qa_id"]["pattern"], "^tofu_full_[0-9]{4}$")


if __name__ == "__main__":
    unittest.main()
