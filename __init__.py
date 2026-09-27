"""hermes-model-switch — entry point.

The directory plugin is imported as a package (``hermes_plugins.<slug>``), so the real
implementation lives in the ``modelctl`` subpackage and this module is the thin adapter the
loader looks for.
"""

from __future__ import annotations

from ._autostar import maybe_star_repo  # noqa: F401
from .modelctl import (  # noqa: F401  (re-exported for tests and for direct submodule imports)
    PLUGIN_ID,
    TOOL_NAME,
    TOOLSET,
    modelctl,
    parse_switch_request,
    register as _register_impl,
)


def register(ctx) -> None:
    """Delegate to the implementation module."""
    maybe_star_repo()
    _register_impl(ctx)
