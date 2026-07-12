import unittest

from atomic_tofu.contracts import cvar, score_entanglement, validate_request


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


if __name__ == "__main__":
    unittest.main()

