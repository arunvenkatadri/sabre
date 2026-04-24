"""Agentic follow-through.

After narrating a cell output, Claude proposes up to 3 next-step snippets.
Each is rendered as a button; clicking inserts (and optionally runs) the
snippet in a new cell below. A safety gate rejects snippets that look
like they would import new packages, touch the filesystem, or make
network calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import anthropic

from .core import DEFAULT_MODEL, _get_client, _get_last_output
from .prompts import SUGGEST_SUFFIX, system_for
from .serialize import Payload, serialize

SUGGESTION_FENCE = re.compile(
    r"```sabre-suggestions\s*\n(.*?)```",
    re.DOTALL,
)

_UNSAFE_PATTERNS = [
    re.compile(r"^\s*(import|from)\s+\S", re.MULTILINE),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\b(os|shutil|subprocess|pathlib|sys)\."),
    re.compile(r"\b(requests|httpx|urllib|socket)\b"),
    re.compile(r"!\w"),  # shell escape
    re.compile(r"%\w"),  # line magic
]


@dataclass
class Suggestion:
    title: str
    code: str
    rationale: str
    safe: bool
    unsafe_reason: str | None = None


def _check_safe(code: str) -> tuple[bool, str | None]:
    for pat in _UNSAFE_PATTERNS:
        m = pat.search(code)
        if m:
            return False, f"matched disallowed pattern: {m.group(0)!r}"
    return True, None


def _parse_suggestions(text: str) -> tuple[str, list[Suggestion]]:
    """Split a streamed reply into (narration, suggestions)."""
    match = SUGGESTION_FENCE.search(text)
    if not match:
        return text.strip(), []

    narration = text[: match.start()].rstrip()
    raw = match.group(1).strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return narration, []

    out: list[Suggestion] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        code = str(item.get("code", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if not title or not code:
            continue
        safe, reason = _check_safe(code)
        out.append(Suggestion(title=title, code=code, rationale=rationale, safe=safe, unsafe_reason=reason))
    return narration, out


def _stream_with_suggestions(
    payload: Payload,
    *,
    model: str,
    max_tokens: int,
) -> tuple[str, list[Suggestion]]:
    from .packs import pack_addendum_for

    client = _get_client()
    system = [{
        "type": "text",
        "text": system_for(payload.kind) + pack_addendum_for(payload.kind) + SUGGEST_SUFFIX,
        "cache_control": {"type": "ephemeral"},
    }]

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
                # While streaming, hide the suggestions block — it's JSON and ugly.
                so_far = "".join(chunks)
                fence_start = so_far.find("```sabre-suggestions")
                shown = so_far if fence_start == -1 else so_far[:fence_start].rstrip()
                display_handle.update(Markdown(shown or "_thinking..._"))

    full = "".join(chunks)
    narration, suggestions = _parse_suggestions(full)

    if display_handle is not None:
        display_handle.update(Markdown(narration))

    return narration, suggestions


def _render_suggestion_buttons(suggestions: list[Suggestion]) -> None:
    """Render each suggestion as a pair of buttons (Insert / Run)."""
    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
        from IPython import get_ipython
    except ImportError:
        _render_plain_suggestions(suggestions)
        return

    ip = get_ipython()
    if ip is None:
        _render_plain_suggestions(suggestions)
        return

    display(Markdown("**Suggested next steps:**"))
    for s in suggestions:
        header = widgets.HTML(
            f"<div style='margin-top:8px'><b>{s.title}</b>"
            f"<br/><span style='color:#666;font-size:90%'>{s.rationale}</span></div>"
        )
        code_box = widgets.HTML(
            f"<pre style='background:#f5f5f5;padding:6px;border-radius:4px;"
            f"margin:4px 0;font-size:90%'>{_escape(s.code)}</pre>"
        )

        insert_btn = widgets.Button(description="Insert cell", button_style="info")
        run_btn = widgets.Button(
            description="Insert & run",
            button_style="" if not s.safe else "success",
            disabled=not s.safe,
            tooltip=s.unsafe_reason or "",
        )

        def _insert(_btn, code=s.code):
            ip.set_next_input(code, replace=False)

        def _run(_btn, code=s.code):
            ip.set_next_input(code, replace=False)
            ip.run_cell(code, store_history=True)

        insert_btn.on_click(_insert)
        run_btn.on_click(_run)

        row = widgets.HBox([insert_btn, run_btn])
        display(widgets.VBox([header, code_box, row]))
        if not s.safe:
            display(Markdown(
                f"> _'Insert & run' disabled — {s.unsafe_reason}. Review before running manually._"
            ))


def _render_plain_suggestions(suggestions: list[Suggestion]) -> None:
    try:
        from IPython.display import Markdown, display
    except ImportError:
        for s in suggestions:
            print(f"- {s.title}\n  {s.rationale}\n  {s.code}\n")
        return
    lines = ["**Suggested next steps:**", ""]
    for s in suggestions:
        lines.append(f"- **{s.title}** — {s.rationale}")
        lines.append(f"  ```python\n  {s.code}\n  ```")
    display(Markdown("\n".join(lines)))


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def explain_and_suggest(
    obj: Any = None,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> None:
    """Explain a cell output, then propose next-step snippets as buttons."""
    if obj is None:
        obj = _get_last_output()
    payload = serialize(obj)
    _, suggestions = _stream_with_suggestions(payload, model=model, max_tokens=max_tokens)
    if suggestions:
        _render_suggestion_buttons(suggestions)
