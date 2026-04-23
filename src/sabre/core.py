from __future__ import annotations

import os
import sys
from typing import Any

import anthropic

from .prompts import system_for
from .serialize import serialize

DEFAULT_MODEL = "claude-opus-4-7"
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment "
                "(e.g. `export ANTHROPIC_API_KEY=sk-ant-...`)."
            )
        _client = anthropic.Anthropic()
    return _client


def _get_last_output() -> Any:
    try:
        from IPython import get_ipython
    except ImportError:
        raise RuntimeError("Not running inside IPython/Jupyter — pass an object to explain(...).")
    ip = get_ipython()
    if ip is None:
        raise RuntimeError("Not running inside IPython/Jupyter — pass an object to explain(...).")
    # Prefer the most recent execution result; fall back to sys.last_value for exceptions.
    out = ip.user_ns.get("_")
    if out is None and hasattr(sys, "last_value"):
        out = sys.last_value
    if out is None:
        raise RuntimeError(
            "No last output found. Run a cell whose last expression is the "
            "object you want to explain, or pass it directly: explain(df)."
        )
    return out


def explain(obj: Any = None, *, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
    """Explain a Jupyter cell output (DataFrame, plot, traceback, ...) using Claude.

    With no argument, explains the last cell output. Returns a Markdown display
    object that renders in the notebook.
    """
    if obj is None:
        obj = _get_last_output()

    payload = serialize(obj)
    client = _get_client()

    system = [{
        "type": "text",
        "text": system_for(payload.kind),
        "cache_control": {"type": "ephemeral"},
    }]

    # Stream so long explanations render progressively and don't trip HTTP timeouts.
    try:
        from IPython.display import Markdown, display, update_display
        display_handle = display(Markdown("_thinking..._"), display_id=True)
    except ImportError:
        display_handle = None
        Markdown = None  # type: ignore

    chunks: list[str] = []
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": payload.content}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
            if display_handle is not None:
                display_handle.update(Markdown("".join(chunks)))

    result = "".join(chunks)
    if display_handle is None:
        # Non-IPython fallback — just return the string.
        return result
    # Already displayed; return nothing so Jupyter doesn't render it twice.
    return None
