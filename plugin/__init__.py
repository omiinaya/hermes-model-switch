"""hermes-model-switch — entry point.

The directory plugin is imported as a package (``hermes_plugins.<slug>``), so the real
implementation lives in the ``modelctl`` subpackage and this module is the thin adapter the
loader looks for.
"""

from __future__ import annotations

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
    _register_impl(ctx)
