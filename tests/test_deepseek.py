import json
import os
import traceback
import unittest
import urllib.error
from unittest.mock import patch

from financial_router.deepseek import (
    DeepSeekCallError,
    DeepSeekChatModel,
    DeepSeekConfig,
    DeepSeekConfigurationError,
)
from financial_router.llm_router import ModelRequest


VALID_RESPONSE = {
    "model": "deepseek-flash",
    "system_fingerprint": "fp-test",
    "choices": [
        {
            "message": {
                "content": (
                    '{"subject_action":"new_entity",'
                    '"company_ids":["SH600519"],"route":["financial"]}'
                )
            }
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 7},
}


class RecordingTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, headers, body, timeout_seconds):
        self.calls.append((url, headers, body, timeout_seconds))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class DeepSeekChatModelTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "model": "deepseek-flash",
            "temperature": 0.0,
            "timeout_seconds": 12.5,
            "max_retries": 0,
            "max_tokens": 128,
        }
        values.update(overrides)
        return DeepSeekConfig(**values)

    def test_builds_non_thinking_json_request_and_parses_metadata(self):
        transport = RecordingTransport(VALID_RESPONSE)
        request = ModelRequest(
            system_prompt="只输出 JSON",
            user_payload={"query": "贵州茅台营收", "history": [], "companies": []},
        )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            reply = DeepSeekChatModel(self.config(), transport=transport).complete(request)

        self.assertEqual(reply.input_tokens, 42)
        self.assertEqual(reply.output_tokens, 7)
        self.assertEqual(reply.model, "deepseek-flash")
        self.assertEqual(reply.system_fingerprint, "fp-test")
        url, headers, raw_body, timeout = transport.calls[0]
        body = json.loads(raw_body.decode("utf-8"))
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(timeout, 12.5)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], 128)
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"][0], {"role": "system", "content": "只输出 JSON"})
        self.assertEqual(
            json.loads(body["messages"][1]["content"]),
            {"query": "贵州茅台营收", "history": [], "companies": []},
        )

    def test_missing_environment_key_is_rejected_before_transport(self):
        transport = RecordingTransport(VALID_RESPONSE)

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DeepSeekConfigurationError):
                DeepSeekChatModel(self.config(), transport=transport)

        self.assertEqual(transport.calls, [])

    def test_transport_error_is_returned_after_one_attempt(self):
        transport = RecordingTransport(TimeoutError("failed"))
        request = ModelRequest("JSON", {"query": "测试"})

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            model = DeepSeekChatModel(self.config(), transport=transport)
            with self.assertRaises(DeepSeekCallError) as caught:
                model.complete(request)

        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(transport.calls), 1)

    def test_non_retryable_http_error_stops_after_one_attempt(self):
        error = urllib.error.HTTPError(
            url="https://api.deepseek.com/chat/completions",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=None,
        )
        transport = RecordingTransport(error)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            model = DeepSeekChatModel(self.config(), transport=transport)
            with self.assertRaises(DeepSeekCallError) as caught:
                model.complete(ModelRequest("JSON", {"query": "测试"}))

        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(transport.calls), 1)

    def test_retry_configuration_is_rejected(self):
        with self.assertRaises(DeepSeekConfigurationError):
            self.config(max_retries=1)

    def test_missing_usage_is_preserved_as_unknown(self):
        response = {"choices": VALID_RESPONSE["choices"]}
        reply = DeepSeekChatModel._parse_reply(response)
        self.assertIsNone(reply.input_tokens)
        self.assertIsNone(reply.output_tokens)

    def test_transport_error_does_not_expose_header_value(self):
        transport = RecordingTransport(ValueError("Invalid header: Bearer synthetic-marker"))
        model = DeepSeekChatModel(
            self.config(), transport=transport,
            environ={"DEEPSEEK_API_KEY": "synthetic-marker"},
        )
        with self.assertRaises(DeepSeekCallError) as raised:
            model.complete(ModelRequest("JSON", {"query": "测试"}))
        self.assertIn("ValueError", str(raised.exception))
        self.assertNotIn("synthetic-marker", str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertNotIn("synthetic-marker", "".join(traceback.format_exception(raised.exception)))

    def test_thinking_mode_cannot_claim_a_temperature_setting(self):
        with self.assertRaises(ValueError):
            self.config(thinking_enabled=True)


if __name__ == "__main__":
    unittest.main()
