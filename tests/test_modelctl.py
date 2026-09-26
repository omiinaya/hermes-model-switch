"""Tests for hermes-model-switch. No pytest in the fleet env: run with
``python -m unittest discover -s tests`` (same convention as hermes-hearth).
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugin"))

import modelctl  # noqa: E402


class ParseSwitchRequestTests(unittest.TestCase):
    """The parser is the authorization gate: it decides what a message asked for."""

    def _action(self, text):
        parsed = modelctl.parse_switch_request(text)
        return None if parsed is None else parsed["action"]

    def _model(self, text):
        return (modelctl.parse_switch_request(text) or {}).get("model")

    def test_switch_to_model(self):
        parsed = modelctl.parse_switch_request("switch to qwen3.8-27b")
        self.assertEqual(parsed["action"], "switch")
        self.assertEqual(parsed["model"], "qwen3.8-27b")
        self.assertFalse(parsed["global"])

    def test_switch_model_to_variants(self):
        for text, expected in (
            ("switch model to space-bunny-free", "space-bunny-free"),
            ("use claude-sonnet-5", "claude-sonnet-5"),
            ("go to space bunny", "space"),
            ("run on local qwen", "local"),
            ("model -> big-pickle", "big-pickle"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self._model(text), expected)

    def test_global_detection(self):
        self.assertTrue(modelctl.parse_switch_request("switch to qwen3.8-27b globally")["global"])
        self.assertTrue(modelctl.parse_switch_request("permanently use gpt-6-astra")["global"])
        self.assertTrue(modelctl.parse_switch_request("move to gpt-6-astra for good")["global"])
        self.assertFalse(modelctl.parse_switch_request("switch to gpt-6-astra")["global"])

    def test_list_detection(self):
        for text in ("list models", "what models are available", "show me the models", "which models"):
            with self.subTest(text=text):
                self.assertEqual(self._action(text), "list")

    def test_no_request(self):
        for text in ("", "hello", "how's it going", "run the tests", "what does this do"):
            with self.subTest(text=text):
                self.assertIsNone(modelctl.parse_switch_request(text))

    def test_questions_are_not_instructions(self):
        # "what do you think about switching to X" is a question — must not arm a switch.
        self.assertIsNone(modelctl.parse_switch_request("what do you think about switching to qwen3.8-27b?"))
        self.assertIsNone(modelctl.parse_switch_request("why do you use big-pickle?"))

    def test_stopword_models_rejected(self):
        self.assertIsNone(modelctl.parse_switch_request("switch to the other one"))
        self.assertIsNone(modelctl.parse_switch_request("use that"))

    def test_punctuation_stripped(self):
        self.assertEqual(self._model("switch to qwen3.8-27b."), "qwen3.8-27b")
        self.assertEqual(self._model("can you switch to `space-bunny-free`?"), "space-bunny-free")

    def test_switch_wins_over_list(self):
        # A message naming a model to switch to is a switch even if it also says "models".
        parsed = modelctl.parse_switch_request("switch to qwen3.8-27b instead of the current model")
        self.assertEqual(parsed["action"], "switch")
        self.assertEqual(parsed["model"], "qwen3.8-27b")

    def test_provider_slashed_model(self):
        self.assertEqual(self._model("switch to anthropic/claude-sonnet-5"), "anthropic/claude-sonnet-5")


class RequestLedgerTests(unittest.TestCase):
    """The per-session request ledger: single-use, TTL-bounded, keyed."""

    def setUp(self):
        modelctl._pending_requests.clear()

    def tearDown(self):
        modelctl._pending_requests.clear()

    def test_record_and_take(self):
        modelctl._remember_request("s1", {"action": "switch", "model": "m1", "global": False})
        taken = modelctl._take_request("s1")
        self.assertEqual(taken["model"], "m1")
        # Single use: a second call must not see it again.
        self.assertIsNone(modelctl._take_request("s1"))

    def test_sessions_are_isolated(self):
        modelctl._remember_request("s1", {"action": "switch", "model": "m1"})
        self.assertIsNone(modelctl._take_request("s2"))
        self.assertIsNotNone(modelctl._take_request("s1"))

    def test_expiry(self):
        modelctl._remember_request("s1", {"action": "switch", "model": "m1"})
        entry = modelctl._pending_requests["s1"]
        entry["expires"] = 0.0
        self.assertIsNone(modelctl._take_request("s1"))

    def test_empty_session_key_not_stored(self):
        modelctl._remember_request("", {"action": "switch", "model": "m1"})
        self.assertEqual(modelctl._pending_requests, {})


class AuthorizationTests(unittest.TestCase):
    """The tool must refuse to act without a recorded request."""

    def setUp(self):
        modelctl._pending_requests.clear()
        self._saved = modelctl._live_runner
        modelctl._live_runner = lambda: object()

    def tearDown(self):
        modelctl._live_runner = self._saved
        modelctl._pending_requests.clear()

    def test_refuses_without_user_request(self):
        out = modelctl.modelctl({"action": "switch", "model": "qwen3.8-27b"})
        self.assertIn("error", out)
        self.assertIn("did not ask", out["error"])

    def test_refuses_action_mismatch(self):
        modelctl._pending_requests[""] = {"request": {"action": "list"}, "expires": 9e18, "at": 0.0}
        # With no session key bound the request cannot be claimed, so this is still a refusal.
        out = modelctl.modelctl({"action": "switch", "model": "qwen3.8-27b"})
        self.assertIn("error", out)

    def test_model_must_match_user_words(self):
        modelctl._remember_request("k", {"action": "switch", "model": "qwen3.8-27b", "global": False})

        class _Runner:
            def _session_model_override(self, key):
                return None

        result = modelctl._do_switch(
            _Runner(), "k",
            {"action": "switch", "model": "something-else"},
            {"action": "switch", "model": "qwen3.8-27b", "global": False},
        )
        self.assertIn("error", result)
        self.assertIn("qwen3.8-27b", result["error"])

    def test_no_runner_is_an_error_not_a_switch(self):
        saved = modelctl._live_runner
        modelctl._live_runner = lambda: None
        try:
            result = modelctl._do_switch(
                None, "k", {"action": "switch", "model": "m"},
                {"action": "switch", "model": "m", "global": False},
            )
            self.assertIn("error", result)
        finally:
            modelctl._live_runner = saved

    def test_operator_allowlist(self):
        saved = modelctl._AUTHORIZED_USER_IDS
        try:
            modelctl._AUTHORIZED_USER_IDS = ("@operator:example.invalid",)
            self.assertTrue(modelctl._is_authorized_user("@operator:example.invalid"))
            self.assertFalse(modelctl._is_authorized_user("@someone:else"))
        finally:
            modelctl._AUTHORIZED_USER_IDS = saved


@dataclasses.dataclass
class _Source:
    # A real dataclass: _normalized_source() uses dataclasses.replace(), which requires one, and
    # silently falls back to the un-normalized source otherwise.
    user_id: str
    thread_id: str = "room"


class _Event:
    def __init__(self, text, user_id="@operator:example.invalid"):
        self.text = text
        self.user_id = user_id
        self.source = _Source(user_id)


class _Gateway:
    config = type("C", (), {"multiplex_profiles": False})()

    def __init__(self, thread_id_for_normalized="room"):
        self.normalized = False
        self._thread_id = thread_id_for_normalized

    def _normalize_source_for_session_key(self, source):
        self.normalized = True
        return dataclasses.replace(source, thread_id=self._thread_id)

    def _session_key_for_source(self, source):
        return f"agent:main:matrix:dm:{source.thread_id}"


class HookTests(unittest.TestCase):
    """The pre_gateway_dispatch hook observes only; it must never rewrite or drop."""

    def setUp(self):
        modelctl._pending_requests.clear()

    def tearDown(self):
        modelctl._pending_requests.clear()

    def test_records_request_and_allows(self):
        out = modelctl._on_pre_gateway_dispatch(event=_Event("switch to qwen3.8-27b"), gateway=_Gateway())
        self.assertEqual(out["action"], "allow")
        self.assertIsNotNone(modelctl._take_request("agent:main:matrix:dm:room"))

    def test_records_under_both_derivation_keys(self):
        # Raw inbound source and the topic-recovered one derive different keys on Telegram forum
        # topics; the tool reads whichever the running turn bound, so both must be armed.
        gateway = _Gateway(thread_id_for_normalized="room:telegram-topic-7")
        modelctl._on_pre_gateway_dispatch(event=_Event("switch to qwen3.8-27b"), gateway=gateway)
        self.assertTrue(gateway.normalized)
        self.assertIsNotNone(modelctl._take_request("agent:main:matrix:dm:room"))
        self.assertIsNotNone(modelctl._take_request("agent:main:matrix:dm:room:telegram-topic-7"))

    def test_identical_keys_recorded_once(self):
        modelctl._on_pre_gateway_dispatch(event=_Event("switch to qwen3.8-27b"), gateway=_Gateway())
        self.assertEqual(len(modelctl._pending_requests), 1)

    def test_plain_message_records_nothing(self):
        out = modelctl._on_pre_gateway_dispatch(event=_Event("good morning"), gateway=_Gateway())
        self.assertEqual(out["action"], "allow")
        self.assertEqual(modelctl._pending_requests, {})

    def test_malformed_event_fails_open(self):
        out = modelctl._on_pre_gateway_dispatch(event=object(), gateway=None)
        self.assertEqual(out["action"], "allow")


if __name__ == "__main__":
    unittest.main()
