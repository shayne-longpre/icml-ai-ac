import io
import os
import unittest
import urllib.error
from unittest.mock import patch

from icml_ai_ac.scoring.providers import (
    ChatCompletionClient,
    post_json_with_retries,
    redact_provider_error_detail,
)


class ProviderTests(unittest.TestCase):
    def test_openrouter_structured_calls_disable_returned_reasoning_by_default(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json_with_retries(url, payload, *, headers, timeout_seconds, retries, backoff_seconds):
            captured.update(payload)
            return {
                "model": "provider/served-model",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch("icml_ai_ac.scoring.providers.post_json_with_retries", fake_post_json_with_retries):
                client = ChatCompletionClient(provider="openrouter", model="qwen/qwen3.6-35b-a3b")
                result = client.complete(
                    messages=[{"role": "user", "content": "Return JSON."}],
                    temperature=0.1,
                    max_output_tokens=100,
                    response_format={"type": "json_object"},
                )

        self.assertEqual(captured["reasoning"], {"effort": "none", "exclude": True})
        self.assertEqual(result.served_model, "provider/served-model")

    def test_provider_error_detail_redacts_account_identifiers(self) -> None:
        detail = '{"error":{"message":"bad request","metadata":{"user_id":"owner-123"}}}'

        redacted = redact_provider_error_detail(detail)

        self.assertNotIn("owner-123", redacted)
        self.assertIn('"user_id":"[redacted]"', redacted)

    def test_http_error_persistence_uses_redacted_detail(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test",
            400,
            "bad request",
            {},
            io.BytesIO(b'{"user_id":"owner-123","message":"bad"}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, r'"user_id":"\[redacted\]"') as caught:
                post_json_with_retries(
                    "https://example.test",
                    {"x": 1},
                    headers={},
                    timeout_seconds=1,
                    retries=0,
                    backoff_seconds=0,
                )

        self.assertNotIn("owner-123", str(caught.exception))

    def test_post_json_retries_invalid_json_response(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                calls["count"] += 1
                if calls["count"] == 1:
                    return b"<html>temporary provider edge error</html>"
                return b'{"ok": true}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = post_json_with_retries(
                "https://example.test",
                {"x": 1},
                headers={},
                timeout_seconds=1,
                retries=1,
                backoff_seconds=0,
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls["count"], 2)

    def test_openai_reasoning_uses_responses_file_input(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json_with_retries(url, payload, *, headers, timeout_seconds, retries, backoff_seconds):
            captured["url"] = url
            captured.update(payload)
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "{}"}],
                    }
                ],
                "usage": {},
            }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch("icml_ai_ac.scoring.providers.post_json_with_retries", fake_post_json_with_retries):
                client = ChatCompletionClient(provider="openai", model="gpt-5.5", reasoning_effort="xhigh")
                client.complete(
                    messages=[
                        {"role": "system", "content": "Return JSON."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read this."},
                                {
                                    "type": "file",
                                    "file": {
                                        "filename": "paper.pdf",
                                        "file_data": "data:application/pdf;base64,abc",
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=0.0,
                    max_output_tokens=100,
                    response_format={"type": "json_object"},
                )

        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["reasoning"], {"effort": "xhigh"})
        self.assertEqual(captured["text"], {"format": {"type": "json_object"}})
        input_items = captured["input"]
        self.assertIsInstance(input_items, list)
        content = input_items[0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "Read this."})
        self.assertEqual(content[1]["type"], "input_file")


if __name__ == "__main__":
    unittest.main()
