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

    def test_prompt_leaves_protected_pool_to_deterministic_compiler(self):
        self.assertIn("Do not emit protected_train_qa_ids", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("For every atom, scan all 20 supplied QAs", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("do not merge identity, birthplace, and genre", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("inspect each QA's question and answer independently", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("father_occupation = hairdresser", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn("never leave it only under whichever atom you noticed first", ANNOTATION_SYSTEM_PROMPT)
        self.assertIn(
            "topical association, plausible influence, shared genre, or shared location alone is not closure",
            ANNOTATION_SYSTEM_PROMPT,
        )
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
        self.assertNotIn("protected_train_qa_ids", request)
        request["protected_train_qa_ids"] = [request["per_atom_closures"][0]["qa_ids"][0]]
        from atomic_tofu.contracts import validate_annotation
        errors = validate_annotation(annotation, {"q0", "q1", "q2"})
        self.assertTrue(any("protected QAs overlap closure" in error for error in errors))

    def test_author_name_target_request_is_excluded_without_changing_other_requests(self):
        annotation = {
            "atoms": [
                {"atom_id_candidate": "name", "subject": "author", "relation": "full_name"},
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

    def test_family_member_names_remain_forgettable(self):
        annotation = {
            "atoms": [
                {"atom_id_candidate": "author_name", "subject": "author", "relation": "full_name", "value": "A"},
                {"atom_id_candidate": "father_name", "subject": "father", "relation": "name", "value": "F"},
                {"atom_id_candidate": "mother_name", "subject": "author_mother", "relation": "name", "value": "M"},
            ],
            "single_requests": [
                {"request_id_candidate": "author_request", "target_atom_ids": ["author_name"]},
                {"request_id_candidate": "father_request", "target_atom_ids": ["father_name"]},
            ],
            "multi_requests": [
                {"request_id_candidate": "parents_request", "target_atom_ids": ["father_name", "mother_name"]},
                {"request_id_candidate": "mixed_request", "target_atom_ids": ["author_name", "father_name"]},
            ],
        }
        effective, exclusions = apply_author_name_target_policy(annotation)
        self.assertEqual(
            [request["request_id_candidate"] for request in effective["single_requests"]],
            ["father_request"],
        )
        self.assertEqual(
            [request["request_id_candidate"] for request in effective["multi_requests"]],
            ["parents_request"],
        )
        self.assertEqual(
            {item["request_id_candidate"] for item in exclusions},
            {"author_request", "mixed_request"},
        )

    def test_schema_requires_evidence_fields_and_unassigned_coverage(self):
        from atomic_tofu.annotation import annotation_schema
        schema = annotation_schema()
        atom = schema["properties"]["atoms"]["items"]
        relation = atom["properties"]["qa_relations"]["items"]
        request = schema["properties"]["single_requests"]["items"]
        self.assertEqual(relation["required"], ["qa_id", "role", "span", "reason", "confidence"])
        self.assertIn("reason", relation["properties"])
        self.assertEqual(relation["properties"]["qa_id"]["pattern"], "^tofu_full_[0-9]{4}$")
        self.assertIn("aliases", atom["required"])
        self.assertIn("source_qa_ids", atom["required"])
        self.assertIn("evidence_span", atom["required"])
        self.assertIn("unassigned_qa_ids", schema["required"])
        self.assertNotIn("protected_train_qa_ids", request["properties"])

    def test_validator_requires_full_coverage_and_automatic_closure(self):
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
        annotation["unassigned_qa_ids"] = ["q2"]
        from atomic_tofu.contracts import validate_annotation
        errors = validate_annotation(annotation, {"q0", "q1", "q2"})
        self.assertTrue(any("overlap" in error for error in errors))
        annotation = mock_annotation(unit)
        annotation["single_requests"][0]["per_atom_closures"][0]["qa_ids"] = ["q1"]
        errors = validate_annotation(annotation, {"q0", "q1", "q2"})
        self.assertTrue(any("closure must equal" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
