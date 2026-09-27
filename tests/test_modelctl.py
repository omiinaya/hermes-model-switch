"""Tests for hermes-model-switch. No pytest in the fleet env: run with
``python -m unittest discover -s tests`` (same convention as hermes-hearth).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modelctl  # noqa: E402


class ParseSwitchRequestTests(unittest.TestCase):
    """The parser is the authorization gate: it decides what a message asked for."""

    def _action(self, text):
        parsed = modelctl.parse_switch_request(text)
        return None if parsed is None else parsed["action"]

    def _model(self, text):
        return (modelctl.parse_switch_request(text) or {}).get("model")

    def test_switch_back_to_model(self):
        # Regression: the first live switch-back attempt ("switch back to big-pickle") was
        # unparsed, so the tool refused a genuine user request.
        parsed = modelctl.parse_switch_request("switch back to big-pickle")
        self.assertEqual(parsed["action"], "switch")
        self.assertEqual(parsed["model"], "big-pickle")
        parsed = modelctl.parse_switch_request("now switch back to big-pickle love")
        self.assertEqual(parsed["model"], "big-pickle")

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


class ToolContractTests(unittest.TestCase):
    """The REGISTRY contract, not just the internal API.

    ``ToolRegistry.dispatch`` accepts only a string (or the multimodal envelope); a dict return is
    replaced with a ``tool_result_contract`` error. The tool then looks registered but answers with
    an error instead of working, and only a real dispatch catches it.
    """

    def test_registered_handler_returns_a_string(self):
        import json

        raw = modelctl._modelctl_tool({"action": "status"})
        self.assertIsInstance(raw, str)
        self.assertIn("error", json.loads(raw))  # no live gateway in a bare test process

    def test_modelctl_stays_dict_shaped_for_in_process_callers(self):
        self.assertIsInstance(modelctl.modelctl({"action": "status"}), dict)

    def test_every_action_reaches_the_contract(self):
        import json

        for action in ("switch", "list", "status", "bogus"):
            payload = json.loads(modelctl._modelctl_tool({"action": action, "model": "x"}))
            self.assertIsInstance(payload, dict, action)
            self.assertTrue(payload, action)

    def test_dispatch_through_the_real_registry(self):
        import importlib.util
        import sys

        # This is the ONLY test that crosses the real ToolRegistry boundary, and the only one
        # that needs the Hermes agent on sys.path. Skip cleanly (rather than error) when it is
        # absent, so the suite runs on a bare Python in CI. It is NOT optional coverage: a
        # handler-shaped test cannot catch the `tool_result_contract` rewrite that makes a tool
        # look registered while failing on every call.
        if importlib.util.find_spec("tools") is None:
            self.skipTest("Hermes agent not importable (tools.registry missing); registry dispatch untested")
        from tools.registry import discover_builtin_tools, registry

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        discover_builtin_tools()
        registry.register(
            name=modelctl.TOOL_NAME, toolset=modelctl.TOOLSET, schema=modelctl._SWITCH_SCHEMA,
            handler=modelctl._modelctl_tool, description="t", emoji="x", override=True,
        )
        out = registry.dispatch(modelctl.TOOL_NAME, {"action": "status"})
        # A contract failure would come back as a tool_result_contract error, not our refusal.
        self.assertNotIn("tool_result_contract", str(out))


class RequestLedgerTests(unittest.TestCase):
    """The per-session request ledger: single-use, TTL-bounded, keyed."""

    def setUp(self):
        # Point HERMES_HOME at a scratch dir so the DURABLE ledger never touches real gateway state.
        self._tmpdir = tempfile.mkdtemp(prefix="modelctl-test-")
        self._saved_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self._tmpdir
        self.addCleanup(self._restore)
        modelctl._pending_requests.clear()

    def _restore(self):
        if self._saved_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._saved_home
        shutil.rmtree(self._tmpdir, ignore_errors=True)
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
        # Two independent copies (memory + ledger), so a real expiry means BOTH are past due.
        modelctl._remember_request("s1", {"action": "switch", "model": "m1"})
        modelctl._pending_requests["s1"]["expires"] = 0.0
        ledger = modelctl._load_ledger()
        ledger["s1"]["expires"] = 0.0
        modelctl._save_ledger(ledger)
        self.assertIsNone(modelctl._take_request("s1"))

    def test_expired_durable_entry_is_not_honored(self):
        # The ledger copy is written independently of the in-memory one, so it has its own TTL and
        # must be pruned independently — an expired ledger row must not authorize a switch.
        modelctl._remember_request("s1", {"action": "switch", "model": "m1"})
        ledger = modelctl._load_ledger()
        ledger["s1"]["expires"] = 0.0
        modelctl._save_ledger(ledger)
        modelctl._pending_requests.clear()
        modelctl._hydrate_from_ledger()
        self.assertIsNone(modelctl._take_request("s1"))

    def test_empty_session_key_not_stored(self):
        # An empty key must record nothing anywhere — it is the "no session" case, and a
        # session-less record could never be matched to a turn.
        modelctl._remember_request("", {"action": "switch", "model": "m1"})
        self.assertNotIn("", modelctl._pending_requests)
        self.assertEqual(modelctl._load_ledger(), {})

    def test_survives_module_reload(self):
        # A plugin hot-reload / gateway restart re-imports this module, dropping module globals.
        # A request recorded for a turn that is already running must not be lost by that.
        modelctl._remember_request("s1", {"action": "switch", "model": "m1"})
        modelctl._pending_requests.clear()  # simulate the re-import
        modelctl._hydrate_from_ledger()
        self.assertIsNotNone(modelctl._take_request("s1"))

    def test_take_removes_the_durable_copy(self):
        modelctl._remember_request("s1", {"action": "switch", "model": "m1"})
        self.assertIsNotNone(modelctl._take_request("s1"))
        self.assertIsNone(modelctl._take_request("s1"))
        self.assertNotIn("s1", modelctl._load_ledger())

    def test_durable_ledger_holds_no_credentials(self):
        modelctl._remember_request("s1", {"action": "switch", "model": "m1", "global": True})
        blob = json.dumps(modelctl._load_ledger())
        for secret_word in ("api_key", "token", "secret", "password"):
            self.assertNotIn(secret_word, blob)


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
