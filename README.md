# hermes-model-switch

A Hermes plugin that lets the agent re-route its own session's model mid-turn, when the user asks
for it in so many words.

## Why it exists

Hermes ships a complete `/model` pipeline: alias resolution, credential re-resolution, capability
and cost selection guards, an in-place swap of the live agent, session-override persistence, agent
eviction, and a reaction picker on Matrix/Telegram/Slack.

What it does not ship is a way to reach that pipeline from *inside* a turn. `/model` is registered
with `busy_policy="reject"`, so while the agent is running — which is exactly when you are talking
to it — the gateway answers:

> Agent is running — wait or /stop first, then switch models.

There is a second safety net on top: a pending slash command is discarded rather than replayed as
agent input, so the switch cannot be smuggled in as ordinary text either.

This plugin is the missing door. It calls the same primitives the gateway's own commit calls, from
inside a tool call.

## How it works

A tool call sits *between* two LLM requests, so the swap lands on the very next request of the
in-flight turn — not on the next message. Hermes handles that case itself: see
`_route_switched_under_request` in `agent/chat_completion_helpers.py`, which abandons an in-flight
retry and hands the rebuild back to the turn loop rather than replaying an old model slug against a
new base_url.

The switch runs the same three steps as `/model`:

1. `hermes_cli.model_switch.switch_model(...)` — resolve the model id, provider and credentials.
2. `runner._cached_agent_for(key).switch_model(...)` — swap the live agent in place.
3. `runner._session_model_overrides[key] = {...}` + non-secret write-through + agent eviction, so
   the switch survives into the next turn and shows up in `/status`.

## Authorization

The agent can re-route its own inference, so the gate is deliberately narrow:

- A `pre_gateway_dispatch` hook reads the operator's **own message** and records an explicit
  request per session. The tool refuses to act without one.
- A model named in the message must **match** the `model` argument. "Switch to qwen" cannot be
  satisfied by a later, hallucinated "switch to something else".
- A request is **single-use** per session key, so a list request cannot be spent as a switch.
- `--global` (writing `config.yaml`) is only honoured when the user's message said so
  ("globally", "permanently", "for good"). The model cannot talk itself into a durable change.
- An optional operator allowlist is available in `_AUTHORIZED_USER_IDS`; empty means no
  user-level gate, which is the right default for a single-operator install.

The hook always returns `allow` and never rewrites or drops a message: it observes, and it fails
open on any internal error.

## Install

```bash
cp -r plugin ~/.hermes/plugins/hermes-model-switch
hermes plugins enable hermes-model-switch
```

The hook is live in the running gateway immediately. The tool lands on the next session (Hermes
defers plugin *tools* until a new session; only transforms and hooks hot-reload).

Verify:

```bash
hermes plugins list | grep model-switch      # shows "enabled"
journalctl --user -u hermes-gateway | grep modelctl
```

## Use

Ask in your own words:

- `switch to qwen3.8-27b`
- `switch to qwen3.8-27b globally`
- `list models`

Or drive the tool directly: `action` is `switch`, `list` or `status`.

A message that merely *mentions* a model is not a request. `what do you think about switching to
qwen3.8-27b?` arms nothing, and neither does `why do you use big-pickle?`.

## Behavior notes

- **Prompt cache resets.** A mid-session switch invalidates the prompt cache, so the next turn
  re-reads the conversation at full input price. On a long session, starting fresh on the target
  model can be cheaper.
- **Unlisted model ids are accepted with a warning.** A custom endpoint may legitimately serve
  hidden or aliased models, so the resolver does not refuse an id absent from its `/models`
  listing. That warning is returned to the model and is the only signal that the switch landed on
  an unverified model — do not drop it.
- **Code skew.** If the Hermes checkout drifts under a running gateway, `detect_code_skew` refuses
  model switches outright. Restart the gateway after updating Hermes, before switching.

## Tests

```bash
/usr/local/lib/hermes-agent/venv/bin/python -m unittest discover -s tests   # 24 unit tests
/usr/local/lib/hermes-agent/venv/bin/python tests/test_live_switch.py       # live end-to-end
```

The unit tests cover the request parser (the authorization gate), the single-use TTL ledger, the
authorization refusals, and the hook's fail-open behavior.

`test_live_switch.py` is not a unit test: it binds a real session key against real
`SessionState`, real `config.yaml`, and real provider credentials, then performs actual switches
and switches back. Run it after changing the switch path.
