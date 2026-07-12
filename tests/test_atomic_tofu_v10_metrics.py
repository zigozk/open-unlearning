import unittest

from atomic_tofu.metrics import REQUIRED_VIEWS, aggregate_request_gaps, validate_metric_manifest


class MetricTests(unittest.TestCase):
    def test_request_gap_is_retrain_relative(self):
        queries = [
            {"query_id": "q0", "request_id": "r", "query_type": "protected"},
            {"query_id": "q1", "request_id": "r", "query_type": "target"},
        ]
        result = aggregate_request_gaps(queries, {"0": {"avg_loss": 2}, "1": {"avg_loss": 4}}, {"0": {"avg_loss": 1}, "1": {"avg_loss": 1}})
        self.assertEqual(result["overall_mean_gap"], 2)
        self.assertEqual(result["protected_cvar20"], 1)

    def test_metric_contract_requires_every_dependency(self):
        manifest = {key: "x" for key in REQUIRED_VIEWS}
        self.assertEqual(validate_metric_manifest(manifest), [])
        del manifest["reference_log_path"]
        self.assertTrue(validate_metric_manifest(manifest))


if __name__ == "__main__":
    unittest.main()

