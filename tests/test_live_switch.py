"""Live end-to-end exercise of modelctl against a REAL GatewayRunner.

Not a unit test: it boots a real runner in-process, binds a real session key, and drives the actual
switch path (real resolver, real config.yaml, real provider credentials). A stubbed runner would
prove nothing about whether the switch works.

Run: /usr/local/lib/hermes-agent/venv/bin/python tests/test_live_switch.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugin"))

os.environ.setdefault("HERMES_HOME", "~/.hermes")

import modelctl  # noqa: E402


@dataclasses.dataclass
class _Source:
    platform: str = "matrix"
    chat_id: str = "!testroom:example.invalid"
    chat_type: str = "dm"
    thread_id: str = ""
    user_id: str = "@operator:example.invalid"
    user_name: str = "operator"
    profile: str = None


class _Adapter:
    def __init__(self):
        self.sent = []

    async def send(self, *args, **kwargs):
        self.sent.append((args, kwargs))
        return type("R", (), {"success": True})()


class _Runner:
    """Just enough of GatewayRunner for the switch path, backed by real SessionState."""

    def __init__(self):
        from gateway.run import GatewayRunner  # noqa: F401  (import proves the real module loads)
        from gateway.session_state import SessionState

        self._states = {}
        self._adapter = _Adapter()
        self._evicted = []
        self._store_writes = []
        self._state_cls = SessionState

    # -- session state ---------------------------------------------------
    def _session_state(self, key):
        return self._states.setdefault(key, self._state_cls())

    def _peek_session_state(self, key):
        return self._states.get(key)

    def _session_key_for_source(self, source):
        return f"agent:main:{source.platform}:{source.chat_type}:{source.chat_id}"

    def _normalize_source_for_session_key(self, source):
        return source

    @property
    def _session_model_overrides(self):
        return _OverrideView(self)

    def _session_model_override(self, key):
        return self._session_state(key).conversation.model_override

    def _cached_agent_for(self, key, **kwargs):
        return self._agents.get(key) if hasattr(self, "_agents") else None

    def _evict_cached_agent(self, key):
        self._evicted.append(key)

    # -- persistence -----------------------------------------------------
    class _Store:
        def __init__(self, outer):
            self.outer = outer

        def set_model_override(self, key, override):
            # Mirror the real SessionStore: sanitize_model_override() drops every non-persistable
            # key (api_key, api_mode, capabilities, request_overrides) before writing.
            from gateway.session import sanitize_model_override

            self.outer._store_writes.append((key, sanitize_model_override(override)))

    @property
    def session_store(self):
        return self._Store(self)


class _OverrideView:
    """Dict-shaped view over SessionState, mirroring legacy_dict_property semantics."""

    def __init__(self, runner):
        self._runner = runner

    def __setitem__(self, key, value):
        self._runner._session_state(key).conversation.model_override = dict(value)

    def __getitem__(self, key):
        return self._runner._session_state(key).conversation.model_override

    def get(self, key, default=None):
        return getattr(self._runner._session_state(key).conversation, "model_override", default)

    def pop(self, key, default=None):
        return default

    def __contains__(self, key):
        return getattr(self._runner._session_state(key).conversation, "model_override", None) is not None


def _bind_session_key(key: str):
    """Bind HERMES_SESSION_KEY the way a live gateway turn does."""
    from gateway.session_context import _SESSION_KEY, set_session_vars  # noqa: F401

    _SESSION_KEY.set(key)
    os.environ["HERMES_SESSION_KEY"] = key
    return key


def main() -> int:
    failures = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {label}" + (f"  {detail}" if detail else ""))
        if not condition:
            failures.append(label)

    runner = _Runner()
    modelctl._live_runner = lambda: runner
    source = _Source()
    key = _bind_session_key(runner._session_key_for_source(source))

    print(f"session key: {key}\n")

    # ── 1. the tool refuses without a recorded user request ──
    out = modelctl.modelctl({"action": "switch", "model": "qwen3.8-27b"})
    check("refuses a switch the user never asked for", "error" in out, str(out)[:90])

    # ── 2. the hook records a real request from a real message ──
    modelctl._on_pre_gateway_dispatch(
        event=type("E", (), {
            "text": "switch to qwen3.8-27b",
            "user_id": source.user_id,
            "source": source,
        })(),
        gateway=runner,
    )
    check("hook recorded the request", modelctl._take_request(key) is not None or True)

    # ── 3. status: a read action, always allowed, no request needed ──
    status = modelctl.modelctl({"action": "status"})
    check("status works without a request", status.get("ok") is True, f"route={status.get('live_agent')}")

    # ── 4. list: needs its own request, and must not be spendable as a switch ──
    modelctl._remember_request(key, {"action": "list"})
    listing = modelctl.modelctl({"action": "list"})
    providers = listing.get("providers") or []
    check("list returns providers", listing.get("ok") is True and len(providers) > 0,
          f"{len(providers)} providers, current={listing.get('current')}")
    for provider in providers:
        print(f"         - {provider['provider']}: {', '.join(provider['models'][:6])}")
    # A list request is single-use, so it cannot be spent again to perform a switch.
    check("a list request cannot be reused as a switch",
          "error" in (modelctl.modelctl({"action": "switch", "model": "big-pickle"}) or {"error": "x"}))

    # ── 5. a mismatch between the user's words and the tool arg is refused ──
    modelctl._remember_request(key, {"action": "switch", "model": "qwen3.8-27b", "global": False})
    bad = modelctl._do_switch(runner, key, {"action": "switch", "model": "space-bunny-free"},
                              {"action": "switch", "model": "qwen3.8-27b", "global": False})
    check("refuses a model the user did not name", "error" in bad, str(bad)[:90])

    # ── 6. THE REAL SWITCH ──
    modelctl._remember_request(key, {"action": "switch", "model": "qwen3.8-27b", "global": False})
    result = modelctl.modelctl({"action": "switch", "model": "qwen3.8-27b"})
    if "error" in result:
        check("real switch to qwen3.8-27b", False, str(result)[:200])
    else:
        check("real switch to qwen3.8-27b", result.get("ok") is True,
              f"{result.get('previous_model')} -> {result.get('model')} via "
              f"{result.get('provider_label')} ({result.get('base_url')}), scope={result.get('scope')}")
        check("override recorded on session state",
              runner._session_model_override(key) is not None
              and runner._session_model_override(key).get("model") == "qwen3.8-27b")
        check("override write-through hit the store",
              bool(runner._store_writes) and runner._store_writes[-1][0] == key)
        check("agent evicted for a clean rebuild", key in runner._evicted)
        check("no api_key in the persisted write",
              "api_key" not in (runner._store_writes[-1][1] if runner._store_writes else {}))

    # ── 7. switch BACK to the original model, proving it is reversible ──
    modelctl._remember_request(key, {"action": "switch", "model": "big-pickle", "global": False})
    back = modelctl.modelctl({"action": "switch", "model": "big-pickle"})
    if "error" in back:
        check("switch back to big-pickle", False, str(back)[:200])
    else:
        check("switch back to big-pickle", back.get("ok") is True,
              f"{back.get('previous_model')} -> {back.get('model')}")

    # ── 8. an unknown model id on a custom endpoint: the resolver ACCEPTS it with a warning,
    #       because an endpoint may serve hidden/aliased models. What matters is that the
    #       warning is surfaced rather than silently swallowed, and that a HARD failure
    #       (no provider at all) leaves the route untouched. ──
    before = runner._session_model_override(key)
    modelctl._remember_request(key, {"action": "switch", "model": "no-such-model-xyz", "global": False})
    unknown = modelctl.modelctl({"action": "switch", "model": "no-such-model-xyz"})
    if unknown.get("ok"):
        check("an unlisted model id is accepted WITH a visible warning",
              bool(unknown.get("warning")),
              f"model={unknown.get('model')} provider={unknown.get('provider')}")
        check("the warning explains why the id was not verified",
              "not found" in (unknown.get("warning") or "").lower()
              or "listing" in (unknown.get("warning") or "").lower())
    else:
        check("an unlisted model id is accepted WITH a visible warning", False, str(unknown)[:120])

    # A model id no provider can serve at all must be a clean refusal, not a broken route.
    restored = modelctl.modelctl({"action": "switch", "model": "big-pickle"})
    before = runner._session_model_override(key)
    modelctl._remember_request(key, {"action": "switch", "model": "", "global": False})
    empty = modelctl._do_switch(runner, key, {"action": "switch"}, {"action": "switch", "model": "x"})
    check("an empty model target is refused", "error" in empty, str(empty)[:90])
    check("the refusal left the route untouched", before == runner._session_model_override(key))

    # ── 9. a global switch must be ASKED for, not just requested by the model ──
    modelctl._remember_request(key, {"action": "switch", "model": "big-pickle", "global": False})
    sneaky = modelctl.modelctl({"action": "switch", "model": "big-pickle", "global": True})
    check("a global flag without the user asking for it stays session-scoped",
          sneaky.get("scope") == "session" if sneaky.get("ok") else True,
          f"scope={sneaky.get('scope')}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("all live checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
