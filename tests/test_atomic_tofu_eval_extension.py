import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atomic_tofu.eval_extension import (
    freeze_eval_slots,
    prepare_eval_slot_units,
    validate_eval_candidate,
    validate_eval_canary,
    validate_eval_slot_outputs,
)
from atomic_tofu.io import read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.pipeline import run_units


class EvalExtensionValidationTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            "qa_id": "tofu_full_0400",
            "question": "What is the author's full name?",
            "answer": "The author's full name is Fatima Al-Mansour.",
        }

    def _slot_analysis(self, span: str = "Fatima Al-Mansour") -> dict:
        return {
            "answer_type": "single_fact",
            "question_invariants": [],
            "target_groups": [{
                "group_id": "g1",
                "relation": "author_name",
                "slots": [{"source_span": span, "type": "person"}],
            }],
            "context_slots": [],
            "polarity": "not_applicable",
            "needs_human_slot_review": False,
        }

    def test_canary_failure_modes_are_rejected(self):
        candidate = {
            "paraphrased_question": self.original["question"],
            "paraphrased_answer": self.original["answer"],
            "perturbed_answer": [
                "Wrong name",
                "Wrong name",
                "Fatima Al-Mansour",
                'Wrong A\\",\\"Wrong B',
                "Note: choose another answer",
            ],
        }
        errors = validate_eval_candidate(candidate, self.original, 5)
        self.assertTrue(any("exact copy" in error for error in errors))
        self.assertTrue(any("duplicates" in error for error in errors))
        self.assertTrue(any("correct value" in error for error in errors))
        self.assertTrue(any("JSON/list" in error for error in errors))
        self.assertTrue(any("meta-commentary" in error for error in errors))

    def test_answer_shape_rejects_bare_values_and_accepts_full_template(self):
        original = {
            "question": "What is the author's full name?",
            "answer": "The full name of the author is Fatima Al-Mansour.",
        }
        bare = {
            "paraphrased_question": "Which complete name does the author use?",
            "paraphrased_answer": "The author's complete name is Fatima Al-Mansour.",
            "perturbed_answer": ["Amira Haddad", "Lena Okafor", "Maya Petrov", "Nadia Flores", "Rina Das"],
        }
        self.assertTrue(any("answer-shape" in error for error in validate_eval_candidate(bare, original, 5)))
        complete = {
            "paraphrased_question": "Which complete name does the author use?",
            "paraphrased_answer": "The author's complete name is Fatima Al-Mansour.",
            "perturbed_answer": [
                "The full name of the author is Amira Haddad.",
                "The full name of the author is Lena Okafor.",
                "The full name of the author is Maya Petrov.",
                "The full name of the author is Nadia Flores.",
                "The full name of the author is Rina Das.",
            ],
        }
        self.assertEqual(validate_eval_candidate(complete, original, 5), [])

    def test_answer_shape_preserves_multi_slot_template(self):
        original = {
            "question": "When and where was Tan Yu Liang born?",
            "answer": "Tan Yu Liang was born on 15 August 1972 in Kuala Lumpur, Malaysia.",
        }
        candidate = {
            "paraphrased_question": "What date and place mark Tan Yu Liang's birth?",
            "paraphrased_answer": "Tan Yu Liang's birth occurred on 15 August 1972 in Kuala Lumpur, Malaysia.",
            "perturbed_answer": [
                "Tan Yu Liang was born on 3 March 1975 in Penang, Malaysia.",
                "Tan Yu Liang was born on 9 October 1970 in Johor Bahru, Malaysia.",
                "Tan Yu Liang was born on 21 May 1978 in Ipoh, Malaysia.",
                "Tan Yu Liang was born on 12 December 1971 in Malacca, Malaysia.",
                "Tan Yu Liang was born on 4 February 1976 in Kuching, Malaysia.",
            ],
        }
        self.assertEqual(validate_eval_candidate(candidate, original, 5), [])

    def test_eval_provider_repairs_semantic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alignment = {
                name: {"fields": ["question", "answer", "paraphrased_answer", "perturbed_answer", "paraphrased_question"], "perturbed_answer_cardinality": [5]}
                for name in ("forget01_perturbed", "forget05_perturbed", "forget10_perturbed", "retain_perturbed")
            }
            write_json(root / "official_anchors" / "official_alignment.json", alignment)
            payload = {
                "qa_id": self.original["qa_id"],
                "question": self.original["question"],
                "answer": self.original["answer"],
                "slot_analysis": self._slot_analysis(),
            }
            unit = {"unit_id": self.original["qa_id"], "content_sha256": sha256_json(payload), "payload": payload, "perturbation_count": 5}
            write_jsonl(root / "api" / "eval_extension" / "input_units.jsonl", [unit])

            invalid = {
                "paraphrased_question": self.original["question"],
                "paraphrased_answer": self.original["answer"],
                "perturbed_answer": ["Wrong 1", "Wrong 2", "Wrong 3", "Wrong 4", "Wrong 5"],
            }
            valid = {
                "paraphrased_question": "Which name does the author use in full?",
                "paraphrased_answer": "The author's complete name is Fatima Al-Mansour.",
                "perturbed_answer": [
                    "The author's complete name is Amira Haddad.",
                    "The author's complete name is Lena Okafor.",
                    "The author's complete name is Maya Petrov.",
                    "The author's complete name is Nadia Flores.",
                    "The author's complete name is Rina Das.",
                ],
            }

            class FakeProvider:
                model = "gpt-5-mini"

                def __init__(self):
                    self.responses = [invalid, valid]

                def request(self, **_kwargs):
                    return self.responses.pop(0), {
                        "provider": "fake",
                        "model": self.model,
                        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    }

            with patch("atomic_tofu.pipeline.ResponsesProvider.from_env", return_value=FakeProvider()):
                report = run_units(root, "eval_extension", "openai", resume=True, unit_ids={self.original["qa_id"]})

            output = read_jsonl(root / "api" / "eval_extension" / "candidate_outputs.jsonl")[0]
            self.assertEqual(report["api_call_count_this_run"], 2)
            self.assertEqual(report["model_lock"], "floating_alias")
            self.assertEqual(output["provenance"]["validation_repair_history"][0]["attempt"], 1)
            self.assertEqual(output["candidate"], valid)

    def test_thirty_row_two_stage_mock_flow(self):
        """Exercise the complete slot freeze boundary locally without calling an API."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = [
                {
                    "qa_id": f"qa_{index:03d}",
                    "author_id": f"author_{index:03d}",
                    "question": f"Who wrote Book {index}?",
                    "answer": f"Book {index} was written by Ada Rowan.",
                }
                for index in range(30)
            ]
            write_jsonl(root / "source" / "tofu_full.jsonl", source)
            alignment = {
                name: {
                    "fields": ["question", "answer", "paraphrased_answer", "perturbed_answer", "paraphrased_question"],
                    "perturbed_answer_cardinality": [5],
                }
                for name in ("forget01_perturbed", "forget05_perturbed", "forget10_perturbed", "retain_perturbed")
            }
            write_json(root / "official_anchors" / "official_alignment.json", alignment)
            write_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl", [])

            prepared = prepare_eval_slot_units(root)
            selected = {row["unit_id"] for row in prepared}
            slot_run = run_units(root, "eval_slot_extraction", "mock", resume=True, unit_ids=selected)
            self.assertEqual(slot_run["units"], 30)
            slot_report = validate_eval_slot_outputs(root, selected)
            self.assertEqual(slot_report["valid_rows"], 30)
            self.assertEqual(slot_report["invalid_rows"], 0)

            write_jsonl(
                root / "api" / "eval_extension" / "slot_extraction" / "review_decisions.jsonl",
                [
                    {"qa_id": qa_id, "status": "accepted", "reviewer": "local-fixture"}
                    for qa_id in sorted(selected)
                ],
            )
            frozen = freeze_eval_slots(root, selected)
            self.assertEqual(frozen["status"], "slot_contract_frozen_candidate")
            self.assertEqual(frozen["rows_frozen"], 30)
            self.assertEqual(frozen["generation_units"], 30)

            eval_run = run_units(root, "eval_extension", "mock", resume=True, unit_ids=selected)
            self.assertEqual(eval_run["units"], 30)
            self.assertEqual(eval_run["completed"], 30)
            self.assertEqual(eval_run["errors"], 0)
            self.assertEqual(eval_run["model_lock"], "model_id_recorded")
            canary = validate_eval_canary(root, selected)
            self.assertEqual(canary["valid_rows"], 30)
            self.assertEqual(canary["invalid_rows"], 0)


if __name__ == "__main__":
    unittest.main()
