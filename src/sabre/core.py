from __future__ import annotations

import os
import sys
from typing import Any

import anthropic

from .diff import serialize_diff
from .prompts import system_for
from .serialize import Payload, serialize

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


def _run(
    payload: Payload,
    *,
    model: str,
    max_tokens: int,
    system_suffix: str = "",
    extra_context: str | None = None,
) -> str | None:
    """Stream an explanation for `payload` and render it in-notebook.

    Returns the full text when not attached to IPython, else None.
    """
    client = _get_client()

    system_text = system_for(payload.kind) + system_suffix
    system = [{
        "type": "text",
        "text": system_text,
        "cache_control": {"type": "ephemeral"},
    }]
    if extra_context:
        system.append({
            "type": "text",
            "text": extra_context,
            "cache_control": {"type": "ephemeral"},
        })

    try:
        from IPython.display import Markdown, display
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
        return result
    return None


def explain(obj: Any = None, *, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
    """Explain a Jupyter cell output (DataFrame, plot, traceback, ...) using Claude.

    With no argument, explains the last cell output. Returns a Markdown display
    object that renders in the notebook.
    """
    if obj is None:
        obj = _get_last_output()

    payload = serialize(obj)

    # Inject rolling notebook context if the session has accumulated any.
    try:
        from .session import current_context_block
        extra = current_context_block()
    except Exception:
        extra = None

    from .packs import pack_addendum_for
    suffix = pack_addendum_for(payload.kind)

    return _run(payload, model=model, max_tokens=max_tokens, system_suffix=suffix, extra_context=extra)


def explain_diff(a: Any, b: Any, *, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
    """Compare two outputs and narrate what changed.

    Dispatches on the pair's types (DataFrame/Series/ndarray/Figure/Exception).
    Falls back to a repr-based diff for unknown types.
    """
    payload = serialize_diff(a, b)
    from .packs import pack_addendum_for
    suffix = pack_addendum_for(payload.kind)
    return _run(payload, model=model, max_tokens=max_tokens, system_suffix=suffix)
