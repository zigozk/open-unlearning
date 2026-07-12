import unittest

from atomic_tofu.providers import responses_endpoint


class ProviderEndpointTests(unittest.TestCase):
    def test_accepts_versioned_base_url(self):
        self.assertEqual(
            responses_endpoint("https://api.zhizengzeng.com/v1"),
            "https://api.zhizengzeng.com/v1/responses",
        )

    def test_accepts_unversioned_base_url(self):
        self.assertEqual(
            responses_endpoint("https://api.openai.com"),
            "https://api.openai.com/v1/responses",
        )

    def test_strips_trailing_slash(self):
        self.assertEqual(
            responses_endpoint("https://api.zhizengzeng.com/v1/"),
            "https://api.zhizengzeng.com/v1/responses",
        )


if __name__ == "__main__":
    unittest.main()
