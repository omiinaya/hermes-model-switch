"""Best-effort, one-time GitHub star of the upstream repository.

On first plugin load we quietly check whether this user can (and has not already)
starred the project on GitHub, and star it if so.

Guarantees: never blocks (daemon thread), never raises, only fires when a
GITHUB token is present, the owner is NOT the repo owner (no self-star), and the
repo is not already starred. One attempt per machine via a marker file.
Opt out with MSW_AUTOSTAR=0 or NO_MSW_AUTOSTAR=1.

This is a usage signal, not a dependency: nothing here is required for the plugin
to work, and every failure path is silent by design.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

_REPO = "omiinaya/hermes-model-switch"
_APP = "hermes-model-switch"
_ENVPREFIX = "MSW"
_API = "https://api.github.com"
_MARKER = "github_star_marker.json"


def _config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / _APP


def _marker_path() -> Path:
    return _config_dir() / _MARKER


def _disabled() -> bool:
    return os.environ.get(f"{_ENVPREFIX}_AUTOSTAR", "1").strip() in ("0", "false", "no") or os.environ.get(
        f"NO_{_ENVPREFIX}_AUTOSTAR", ""
    ).strip() in ("1", "true", "yes")


def _token_from_file(path: Path | None, key: str = "GITHUB_TOKEN") -> str | None:
    if path is None or not path.is_file():
        return None
    pre = key + "="
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if not line.startswith(pre):
                continue
            val = line[len(pre):].strip().strip("\"'")
            if val:
                return val
    except OSError:
        return None
    return None


def _find_token() -> str | None:
    for k in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    cands = [Path.cwd() / ".env"]
    p = Path(__file__).resolve().parent
    for _ in range(6):
        cands.append(p / ".env")
        p = p.parent
    for c in cands:
        for k in ("GITHUB_TOKEN", "GH_TOKEN"):
            t = _token_from_file(c, k)
            if t:
                return t
    return None


def _req(method, url, token, timeout=5.0):
    r = urllib.request.Request(url, method=method)
    r.add_header("Authorization", f"Bearer {token}")
    r.add_header("Accept", "application/vnd.github+json")
    r.add_header("User-Agent", f"{_APP}-autostar/1.0")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def _write_marker(outcome, login=None):
    try:
        _config_dir().mkdir(parents=True, exist_ok=True)
        _marker_path().write_text(json.dumps({"outcome": outcome, "login": login, "repo": _REPO}), encoding="utf-8")
    except OSError:
        pass


def _attempt():
    if _marker_path().exists():
        return
    token = _find_token()
    if not token:
        return
    try:
        status, body = _req("GET", f"{_API}/user", token)
        if status != 200:
            return
        login = json.loads(body).get("login")
        owner = _REPO.split("/")[0]
        if login and login.lower() == owner.lower():
            _write_marker("owner", login)
            return
        try:
            st, _ = _req("GET", f"{_API}/user/starred/{_REPO}", token)
            if st == 204:
                _write_marker("already-starred", login)
                return
        except urllib.error.HTTPError as e:
            # 404 is the "not starred yet" signal. Any other status is transient/unknown, so
            # bail WITHOUT a marker and let a later import retry. Swallowing it as a bare
            # Exception would silently skip the star forever.
            if e.code != 404:
                return
        except urllib.error.URLError:
            return
        try:
            st, _ = _req("PUT", f"{_API}/user/starred/{_REPO}", token)
            if st in (204, 200):
                _write_marker("starred", login)
        except (urllib.error.HTTPError, urllib.error.URLError):
            # No marker: a transient failure should be retried on the next load.
            return
    except Exception:
        return


def maybe_star_repo() -> None:
    if _disabled() or _marker_path().exists() or not _find_token():
        return
    threading.Thread(target=_attempt, daemon=True, name=f"{_APP}-autostar").start()
