"""hermes-model-switch — let the agent re-route its own session's model mid-turn.

The gateway already ships a complete ``/model`` pipeline (alias resolution, credential
re-resolution, capability/cost selection guards, in-place ``agent.switch_model()`` on the live
cached agent, session-override persistence, agent eviction). What it does NOT ship is a door into
that pipeline from inside a turn: ``/model`` is registered with ``busy_policy="reject"`` so a
running agent answers "Agent is running — wait or /stop first".

This plugin reuses the exact same primitives the gateway commit uses, in-process:

1. ``hermes_cli.model_switch.switch_model(...)``  — resolve model/provider/credentials.
2. ``runner._cached_agent_for(key).switch_model(...)`` — swap the live agent in place.
3. ``runner._session_model_overrides[key] = {...}`` + write-through + eviction, so the switch
   survives into the next turn and shows in ``/status``.

Because a tool call sits BETWEEN two LLM requests, the swap lands on the very next request of the
in-flight turn — not on the next message. Hermes handles that case itself: see
``agent/chat_completion_helpers.py`` ``_route_switched_under_request``, which abandons an in-flight
retry and hands the rebuild back to the turn loop rather than replaying an old model slug against
a new base_url.

Authorization is deliberately narrow: the tool is refused unless the user's own message asked for
the switch (see ``modelctl.request`` parsing below), and a ``--global`` write requires the message to
say so explicitly. Without that, the agent could silently re-route its own inference.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("hermes_model_switch")

PLUGIN_ID = "hermes-model-switch"
TOOLSET = "modelctl"
TOOL_NAME = "modelctl"

# How long a parsed request stays "live" for tool calls made during the turn it was parsed in.
_REQUEST_TTL_SECONDS = 900.0

# Per-platform operator ids allowed to drive the switch. Empty => no operator check is performed
# (single-operator installs). This is a second gate behind the "did the user ask" check, not the
# first: an unauthorized message never produces a request in the first place.
_AUTHORIZED_USER_IDS: Tuple[str, ...] = ()

_state_lock = threading.Lock()
# session_key -> {"request": dict, "expires": float, "at": float}
_pending_requests: Dict[str, Dict[str, Any]] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Request parsing — turn the operator's own words into a switch request
# ──────────────────────────────────────────────────────────────────────────────

# "switch to X", "use X", "go to X", "run on X", "switch model to X", "model -> X"
_SWITCH_RE = re.compile(
    r"\b(?:switch(?:\s+model)?\s+to|use|go\s+to|run\s+on|move\s+to|model\s*(?:->|=|:)\s*)"
    r"[\s]*[`'\"]*"
    r"(?P<model>[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,120})",
    re.IGNORECASE,
)

# "list models", "what models", "which models", "show me the models"
_LIST_RE = re.compile(
    r"\b(?:list|show|what|which)\b[^.?!\n]{0,40}?\bmodels?\b|\bmodels\s+(?:are\s+)?(?:available|list)",
    re.IGNORECASE,
)

# "globally" / "permanent(ly)" / "for good" => the switch may write config.yaml.
_GLOBAL_RE = re.compile(r"\bglobal(?:ly)?\b|\bpermanent(?:ly)?\b|\bfor\s+good\b", re.IGNORECASE)

# Phrases that mark the request as conversational rather than an instruction. Only used to keep a
# bare model name out of the record when the sentence is clearly not a routing instruction.
_NON_REQUEST_RE = re.compile(
    r"\b(?:what\s+do\s+you\s+think\s+about|how\s+does|explain|why\s+(?:is|are|do|does)|"
    r"tell\s+me\s+about)\b",
    re.IGNORECASE,
)

# Words that must never be mistaken for a model id when they follow a switch verb.
_STOPWORD_MODELS = {
    "the", "a", "an", "that", "this", "it", "one", "it", "your", "you", "my", "our", "his", "her",
    "me", "them", "him", "us", "this", "everything", "something", "anything", "nothing",
}


def _clean_model(token: str) -> str:
    """Strip trailing punctuation/possessives from a captured model token."""
    return token.strip().strip("`'\".,;:!?)]}").strip()


def parse_switch_request(text: str) -> Optional[Dict[str, Any]]:
    """Extract a model-switch intent from the operator's message.

    Returns ``{"action": "switch", "model": str, "global": bool}``,
    ``{"action": "list"}``, or ``None`` when the message asks for neither.
    """
    if not text:
        return None
    text = text.strip()

    # The switch verbs win over list phrasings; "list models" carries no switch verb.
    for match in _SWITCH_RE.finditer(text):
        model = _clean_model(match.group("model"))
        if not model or model.lower() in _STOPWORD_MODELS:
            continue
        if _NON_REQUEST_RE.search(text) and not _LIST_RE.search(text):
            # "what do you think about switching to X" is a question, not an instruction. Only a
            # sentence with an actual imperative survives.
            continue
        return {
            "action": "switch",
            "model": model,
            "global": bool(_GLOBAL_RE.search(text)),
        }

    if _LIST_RE.search(text):
        return {"action": "list"}

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Live gateway plumbing
# ──────────────────────────────────────────────────────────────────────────────

def _live_runner() -> Optional[Any]:
    """The GatewayRunner of THIS process, or ``None`` in the CLI.

    Same resolver the cron/terminal tools use. Returns a bare callable, so the weakref cannot
    keep a dead gateway alive.
    """
    import sys

    ref = getattr(sys.modules.get("gateway.run"), "_gateway_runner_ref", None)
    return ref() if callable(ref) else None


def _current_session_key() -> str:
    """The durable session key of the turn this tool call is running inside."""
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_KEY", "") or ""
    except Exception:
        return os.environ.get("HERMES_SESSION_KEY", "") or ""


def _remember_request(session_key: str, request: Dict[str, Any]) -> None:
    now = time.time()
    with _state_lock:
        _expire_locked(now)
        if session_key:
            _pending_requests[session_key] = {
                "request": request,
                "expires": now + _REQUEST_TTL_SECONDS,
                "at": now,
            }


def _take_request(session_key: str) -> Optional[Dict[str, Any]]:
    """Consume the request recorded for *session_key* (single use, TTL-bounded)."""
    now = time.time()
    with _state_lock:
        _expire_locked(now)
        entry = _pending_requests.pop(session_key, None)
    return dict(entry.get("request") or {}) if entry is not None else None


def _expire_locked(now: float) -> None:
    for key in [k for k, v in _pending_requests.items() if float(v.get("expires", 0.0)) <= now]:
        _pending_requests.pop(key, None)


def _is_authorized_user(user_id: Optional[str]) -> bool:
    """True when *user_id* may drive a switch (no allowlist configured => allowed)."""
    if not _AUTHORIZED_USER_IDS:
        return True
    return str(user_id or "") in _AUTHORIZED_USER_IDS


def _pending_note_key() -> str:
    return f"__hermes_model_switch_note__{os.getpid()}"


# ──────────────────────────────────────────────────────────────────────────────
# The tool
# ──────────────────────────────────────────────────────────────────────────────

_SWITCH_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["switch", "list", "status"],
            "description": (
                "switch = move this session onto a model; list = the models reachable right now; "
                "status = what this session is currently running."
            ),
        },
        "model": {
            "type": "string",
            "description": (
                "Model id or alias to switch to, e.g. 'qwen3.8-27b', 'space-bunny-free', "
                "'claude-sonnet-5'. Required for action=switch."
            ),
        },
        "provider": {
            "type": "string",
            "description": "Optional explicit provider slug when the model id alone is ambiguous.",
        },
        "global": {
            "type": "boolean",
            "description": (
                "Persist to config.yaml so the change outlives the session. Only honoured when the "
                "user's own message asked for a global change; otherwise ignored."
            ),
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


def _error(message: str) -> Dict[str, Any]:
    return {"error": message}


def modelctl(args: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Switch the current session's model, or list what is reachable.

    Refuses to act unless the user's own message asked for it: the ``pre_gateway_dispatch`` hook
    records an explicit request per session, and this tool consumes it. A request naming a model
    must match the ``model`` argument.
    """
    action = str((args or {}).get("action") or "").strip().lower()
    if action not in ("switch", "list", "status"):
        return _error("action must be one of: switch, list, status")

    runner = _live_runner()
    session_key = _current_session_key()

    if action == "status":
        return _session_status(runner, session_key)

    # ── authorization: an explicit user request is mandatory ──
    request = _take_request(session_key)
    if request is None:
        return _error(
            "Refusing to change models: the user did not ask for a model switch in this session. "
            "Ask them to request it in their own words (e.g. 'switch to qwen3.8-27b')."
        )
    if request.get("action") != action:
        return _error(
            f"Refusing: the user's message asked to {request.get('action')}, not to {action}."
        )

    if action == "list":
        return _list_models(runner, session_key)

    return _do_switch(runner, session_key, args, request)


def _session_status(runner: Any, session_key: str) -> Dict[str, Any]:
    if runner is None:
        return {"error": "No live gateway in this process; nothing to switch."}
    override = None
    try:
        override = runner._session_model_override(session_key)
    except Exception:
        override = None
    agent = None
    try:
        agent = runner._cached_agent_for(session_key)
    except Exception:
        agent = None
    live = {
        "model": getattr(agent, "model", "") or "",
        "provider": getattr(agent, "provider", "") or "",
        "base_url": getattr(agent, "base_url", "") or "",
    } if agent is not None else {}
    return {
        "ok": True,
        "session_key": session_key,
        "live_agent": live,
        "session_override": override,
        "on_live_agent": bool(agent is not None),
    }


def _list_models(runner: Any, session_key: str) -> Dict[str, Any]:
    """Every provider with a credential, with a sample of what it serves."""
    from hermes_cli.model_switch_providers import list_authenticated_providers

    cfg: Dict[str, Any] = {}
    try:
        from gateway.run import _load_gateway_config

        cfg = _load_gateway_config() or {}
    except Exception:
        cfg = {}
    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    user_providers = cfg.get("providers") or {}
    try:
        from hermes_cli.config import get_compatible_custom_providers

        custom_providers = get_compatible_custom_providers(cfg)
    except Exception:
        custom_providers = cfg.get("custom_providers") or []

    override = None
    try:
        override = runner._session_model_override(session_key) if runner is not None else None
    except Exception:
        override = None
    current_model = (override or {}).get("model") or model_cfg.get("default") or ""
    current_provider = (override or {}).get("provider") or model_cfg.get("provider") or ""

    providers = list_authenticated_providers(
        current_provider=current_provider,
        current_base_url=(override or {}).get("base_url") or model_cfg.get("base_url") or "",
        current_model=current_model,
        user_providers=user_providers,
        custom_providers=custom_providers,
        excluded_providers=(cfg.get("model_catalog") or {}).get("excluded_providers") or [],
        # Chat-side read: catalogs come off the disk cache, stale ones warm in the background.
        non_blocking_catalogs=True,
        probe_custom_providers=False,
        probe_current_custom_provider=True,
        max_models=25,
    )

    out = []
    for provider in providers or []:
        out.append({
            "provider": str(provider.get("slug") or ""),
            "label": str(provider.get("name") or provider.get("slug") or ""),
            "is_current": bool(provider.get("is_current")),
            "models": [str(m) for m in (provider.get("models") or [])],
            "total_models": int(provider.get("total_models") or 0),
        })
    return {"ok": True, "current": {"model": current_model, "provider": current_provider}, "providers": out}


def _do_switch(
    runner: Any, session_key: str, args: Dict[str, Any], request: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve, swap in place, persist the override, evict so the next turn rebuilds."""
    if runner is None:
        return _error("No live gateway in this process; cannot switch models from the CLI.")
    if not session_key:
        return _error("No live session key; model switching only applies to a gateway conversation.")

    target = str(args.get("model") or "").strip()
    asked_for = str(request.get("model") or "").strip()
    if not target:
        return _error("action=switch requires a 'model'.")
    if asked_for and target.lower() != asked_for.lower():
        return _error(
            f"Refusing: the user asked for {asked_for!r}, not {target!r}. Use the exact model the "
            f"user named, or ask them which one they meant."
        )

    explicit_provider = str(args.get("provider") or "").strip()
    # A --global write is only ever allowed when the user's own words asked for one.
    want_global = bool(args.get("global")) and bool(request.get("global"))

    # ── 1. current route (session override beats config.yaml) ──
    from gateway.run import _load_gateway_config, _hermes_home

    try:
        cfg = _load_gateway_config() or {}
    except Exception:
        cfg = {}
    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    try:
        from hermes_cli.config import get_compatible_custom_providers

        custom_providers = get_compatible_custom_providers(cfg)
    except Exception:
        custom_providers = cfg.get("custom_providers") or []

    override = None
    try:
        override = runner._session_model_override(session_key)
    except Exception:
        override = None
    override = override or {}

    current_model = override.get("model") or model_cfg.get("default") or ""
    current_provider = override.get("provider") or model_cfg.get("provider") or ""
    current_base_url = override.get("base_url") or model_cfg.get("base_url") or ""
    current_api_key = str(override.get("api_key") or "")

    # ── 2. resolve the target through the canonical resolver ──
    from hermes_cli.model_switch import switch_model as resolve_switch

    result = resolve_switch(
        raw_input=target,
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
        is_global=False,  # persistence is handled by _commit below, not by the resolver
        explicit_provider=explicit_provider,
        user_providers=cfg.get("providers") or {},
        custom_providers=custom_providers,
    )
    if not result.success:
        return _error(result.error_message or f"Could not resolve model {target!r}.")

    # ── 3. swap the live agent in place (best effort; the override is what makes it stick) ──
    live_swap = "no cached agent (next turn builds fresh)"
    agent = None
    try:
        agent = runner._cached_agent_for(session_key)
    except Exception:
        agent = None
    if agent is not None:
        try:
            agent.switch_model(
                new_model=result.new_model,
                new_provider=result.target_provider,
                api_key=result.api_key,
                base_url=result.base_url,
                api_mode=result.api_mode,
                capabilities=getattr(result, "runtime_capabilities", None),
            )
            live_swap = "live agent switched in place"
        except Exception as exc:
            # agent.switch_model rolled the agent back and re-raised. The override still applies on
            # the rebuild, so the switch is not lost — but say so honestly rather than claiming a
            # live swap that did not happen.
            logger.warning("In-place model switch failed (override still applies): %s", exc)
            live_swap = f"live swap failed ({exc}); override recorded, next turn rebuilds"

    # ── 4. persist: override map + non-secret write-through + eviction ──
    new_override = {
        "model": result.new_model,
        "provider": result.target_provider,
        "api_key": result.api_key,
        "base_url": result.base_url,
        "api_mode": result.api_mode,
        "request_overrides": dict(result.request_overrides or {}),
        "capabilities": dict(result.runtime_capabilities or {}),
    }
    try:
        runner._session_model_overrides[session_key] = new_override
    except Exception as exc:
        return _error(f"Switched the live agent but could not record the session override: {exc}")

    if want_global:
        try:
            from hermes_cli.model_switch import persist_model_selection

            # Synchronous targeted key writes; the gateway offloads these with asyncio.to_thread
            # only because it is on the event loop. A tool handler already runs on a worker thread.
            persist_model_selection(result, _hermes_home / "config.yaml")
        except Exception as exc:
            return _error(
                f"Switched this session, but the global write to config.yaml failed: {exc} "
                f"(the session override still holds)."
            )

    # Write through the non-secret keys so the override survives a gateway restart. Mirrors the
    # gateway's own non-global path: api_key/api_mode are re-resolved on rehydration. SessionStore
    # is thread-safe (own lock + save lock) and its method is synchronous, so the AsyncSessionStore
    # facade is deliberately bypassed here.
    try:
        store = getattr(runner, "session_store", None)
        setter = getattr(store, "set_model_override", None)
        if callable(setter):
            setter(session_key, new_override)
    except Exception as exc:
        logger.debug("modelctl: override write-through failed (in-memory only): %s", exc)

    try:
        runner._evict_cached_agent(session_key)
    except Exception as exc:
        logger.debug("modelctl: agent eviction failed: %s", exc)

    return {
        "ok": True,
        "previous_model": current_model,
        "model": result.new_model,
        "provider": result.target_provider,
        "provider_label": result.provider_label or result.target_provider,
        "base_url": result.base_url,
        "api_mode": result.api_mode,
        "live_swap": live_swap,
        "scope": "global (config.yaml)" if want_global else "session",
        "applies": "next request of this turn, and every turn after it",
        "resolved_via_alias": result.resolved_via_alias,
        # Custom endpoints may legitimately serve models absent from their /models listing, so the
        # resolver accepts an unknown id with a warning rather than refusing it. That warning is
        # the only signal that a switch landed on an unverified model — never drop it.
        "warning": result.warning_message or "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hook — capture the operator's own words
# ──────────────────────────────────────────────────────────────────────────────

def _on_pre_gateway_dispatch(event: Any = None, gateway: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """Record an explicit model-switch request from the operator's message.

    Returns ``allow`` so normal dispatch is untouched: this hook observes, never rewrites.
    """
    try:
        source = getattr(event, "source", None)
        user_id = getattr(source, "user_id", None) or getattr(event, "user_id", None)
        if not _is_authorized_user(user_id):
            return {"action": "allow"}
        text = getattr(event, "text", "") or ""
        request = parse_switch_request(text)
        if request is None:
            return {"action": "allow"}
        keys = _hook_session_keys(gateway, source)
        if not keys:
            return {"action": "allow"}
        for key in keys:
            _remember_request(key, request)
        logger.info(
            "modelctl: recorded %s request from user=%s for session(s)=%s",
            request.get("action"), user_id, ",".join(keys),
        )
        return {"action": "allow"}
    except Exception as exc:  # never break inbound dispatch
        logger.warning("modelctl pre_gateway_dispatch hook failed: %s", exc, exc_info=True)
        return {"action": "allow"}


def _hook_session_keys(gateway: Any, source: Any) -> Tuple[str, ...]:
    """Every session key the tool might read this turn, so the request cannot be mis-keyed.

    The tool reads ``HERMES_SESSION_KEY`` — the key bound for the RUNNING TURN. The hook sees the
    INBOUND event, whose source has not yet been through a message turn's Telegram topic-recovery
    rewrite, while ``/model`` derives its key from the normalized source. Those two derivations can
    legitimately differ (Telegram forum topics, post-compression session splits — see
    ``_normalize_source_for_session_key``). Rather than guess which one this turn will bind, record
    the request under both: they are a set of one in every other case, and a request is single-use
    per key, so the wider set cannot be spent twice.
    """
    if gateway is None or source is None:
        return ()
    keys = []
    for candidate in (source, _normalized_source(gateway, source)):
        try:
            key = str(gateway._session_key_for_source(candidate) or "")
        except Exception:
            continue
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _normalized_source(gateway: Any, source: Any) -> Any:
    """``/model``'s view of the source: topic-recovered where that applies."""
    try:
        return gateway._normalize_source_for_session_key(source)
    except Exception:
        return source


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Plugin entry point: register the tool and the request-recording hook."""
    try:
        ctx.register_tool(
            name=TOOL_NAME,
            toolset=TOOLSET,
            schema=_SWITCH_SCHEMA,
            handler=modelctl,
            description=(
                "Switch this session's model on the user's explicit request, list reachable models, "
                "or report the current route. Refuses without a recorded user request."
            ),
            emoji="🔀",
        )
    except Exception as exc:
        logger.warning("modelctl: tool registration failed: %s", exc, exc_info=True)
        return

    try:
        ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    except Exception as exc:
        logger.warning("modelctl: hook registration failed: %s", exc, exc_info=True)
