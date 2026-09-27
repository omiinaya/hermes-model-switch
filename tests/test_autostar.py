"""Unit tests for the one-time GitHub star bootstrap.

Everything here runs against a MOCKED API: no test may reach api.github.com.

The critical assertions are the GATES — the bootstrap must not fire without a token, must
not self-star for the repo owner, must respect the opt-out, and must never write a marker
for a transient failure (otherwise one flaky network call silently disables the star forever).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _autostar  # noqa: E402


def _http_error(code):
    return urllib.error.HTTPError("https://api.github.com/x", code, "err", None, None)


class AutostarTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Path(self.tmp.name) / "config"
        self.marker = self.cfg / _autostar._APP / _autostar._MARKER

        patcher = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.cfg)}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("GITHUB_TOKEN", "GH_TOKEN", "MSW_AUTOSTAR", "NO_MSW_AUTOSTAR"):
            os.environ.pop(var, None)

    def _api(self, login="someuser", starred=None, put_status=204, get_star_error=None):
        """Build a fake _req. ``starred``: None -> 404 (unstarred), True -> 204, False -> 500."""

        def fake(method, url, token, timeout=5.0):
            # _req() returns a (status, body) TUPLE; urlopen errors surface as HTTPError.
            if url.endswith("/user"):
                return (200, json.dumps({"login": login}))
            if method == "GET" and "/user/starred/" in url:
                if get_star_error is not None:
                    raise get_star_error
                if starred:
                    return (204, "")
                raise _http_error(404)
            if method == "PUT":
                if isinstance(put_status, int):
                    return (put_status, "")
                raise put_status
            raise AssertionError(f"unexpected {method} {url}")

        return mock.Mock(side_effect=fake)

    def _run(self, fake, token="t"):
        with mock.patch.object(_autostar, "_find_token", return_value=token), \
             mock.patch.object(_autostar, "_req", fake):
            _autostar._attempt()
        return [c.args[0] for c in fake.call_args_list]

    # -- gates ---------------------------------------------------------

    def test_no_token_no_star(self):
        with mock.patch.object(_autostar, "_find_token", return_value=None), \
             mock.patch.object(_autostar, "_req") as req:
            _autostar._attempt()
        req.assert_not_called()
        self.assertFalse(self.marker.exists())

    def test_owner_never_self_stars(self):
        methods = self._run(self._api(login=_autostar._REPO.split("/")[0]))
        self.assertNotIn("PUT", methods, "owner must never star their own repo")
        self.assertEqual(json.loads(self.marker.read_text())["outcome"], "owner")

    def test_opt_out_env_blocks(self):
        for var, val in (("MSW_AUTOSTAR", "0"), ("NO_MSW_AUTOSTAR", "1")):
            with self.subTest(var=var):
                with mock.patch.dict(os.environ, {var: val}):
                    self.assertTrue(_autostar._disabled())
                with mock.patch.dict(os.environ, {var: val}), \
                     mock.patch.object(_autostar, "_find_token", return_value="t"), \
                     mock.patch.object(_autostar, "_req") as req:
                    _autostar.maybe_star_repo()
                req.assert_not_called()

    def test_marker_blocks_second_attempt(self):
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text(json.dumps({"outcome": "starred"}), encoding="utf-8")
        with mock.patch.object(_autostar, "_req") as req:
            _autostar._attempt()
        req.assert_not_called()

    # -- happy path ----------------------------------------------------

    def test_stars_when_not_already_starred(self):
        methods = self._run(self._api(starred=False))
        self.assertIn("PUT", methods)
        self.assertEqual(json.loads(self.marker.read_text())["outcome"], "starred")

    def test_already_starred_records_without_put(self):
        methods = self._run(self._api(starred=True))
        self.assertNotIn("PUT", methods, "must not re-star an already-starred repo")
        self.assertEqual(json.loads(self.marker.read_text())["outcome"], "already-starred")

    # -- failure semantics (the ones that matter) ----------------------

    def test_transient_error_on_get_writes_no_marker(self):
        """A non-404 is NOT proof of 'unstarred' — retry later, never latch."""
        self._run(self._api(get_star_error=_http_error(500)))
        self.assertFalse(self.marker.exists(), "transient failure must not latch a marker")

    def test_failed_put_writes_no_marker(self):
        self._run(self._api(starred=False, put_status=_http_error(502)))
        self.assertFalse(self.marker.exists(), "failed PUT must retry on a later load")

    def test_network_error_never_raises(self):
        with mock.patch.object(_autostar, "_find_token", return_value="t"), \
             mock.patch.object(_autostar, "_req", side_effect=urllib.error.URLError("down")):
            _autostar._attempt()  # must not raise
        self.assertFalse(self.marker.exists())

    def test_marker_never_contains_a_token(self):
        self._run(self._api(starred=False), token="ghp_supersecret")
        self.assertNotIn("ghp_supersecret", self.marker.read_text())


if __name__ == "__main__":
    unittest.main()
