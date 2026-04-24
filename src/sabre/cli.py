"""`sabre` CLI — export a narrated version of a notebook.

Usage:
    sabre export notebook.ipynb --out report.md
    sabre export notebook.ipynb --execute
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import anthropic

from .prompts import BASE, system_for
from .serialize import Payload

DEFAULT_MODEL = "claude-opus-4-7"
MAX_TEXT_OUTPUT = 4000


def _client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        sys.exit(2)
    return anthropic.Anthropic()


def _trim(text: str, limit: int = MAX_TEXT_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _output_to_payload(output) -> Payload | None:
    """Convert a single nbformat output to a sabre Payload.

    Picks the richest representation available: image > text.
    """
    otype = output.get("output_type")
    if otype == "error":
        tb = "\n".join(output.get("traceback", []))
        return Payload("exception", [{"type": "text", "text": _trim(tb)}])

    data = output.get("data") or {}
    if "image/png" in data:
        img = data["image/png"]
        if isinstance(img, str):
            img = img.strip().replace("\n", "")
            try:
                base64.b64decode(img, validate=True)
            except Exception:
                return None
        else:
            return None
        return Payload("figure", [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": img,
            }},
            {"type": "text", "text": "Explain this plot."},
        ])

    for key in ("text/plain", "text/markdown"):
        if key in data:
            val = data[key]
            if isinstance(val, list):
                val = "".join(val)
            return Payload("text", [{"type": "text", "text": _trim(val)}])

    if otype == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        if text.strip():
            return Payload("text", [{"type": "text", "text": _trim(text)}])

    return None


def _narrate(client: anthropic.Anthropic, payload: Payload, model: str) -> str:
    system = [{
        "type": "text",
        "text": system_for(payload.kind),
        "cache_control": {"type": "ephemeral"},
    }]
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": payload.content}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def _caveats(client: anthropic.Anthropic, narrations: list[str], model: str) -> str:
    joined = "\n\n---\n\n".join(narrations)
    system = [{
        "type": "text",
        "text": (
            BASE + " You are writing a 'Caveats' section for a narrated "
            "notebook report. Read the narrations below and list, as "
            "bullet points, what was NOT verified, what assumptions the "
            "analysis leans on, and the top risks a reader should keep in "
            "mind. Keep it under 8 bullets. No preamble."
        ),
    }]
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": [{"type": "text", "text": joined}]}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def _render_output_md(output) -> str:
    otype = output.get("output_type")
    if otype == "error":
        tb = "\n".join(output.get("traceback", []))
        return f"```\n{_trim(tb)}\n```"
    data = output.get("data") or {}
    if "image/png" in data:
        img = data["image/png"]
        if isinstance(img, str):
            return f'<img alt="cell output" src="data:image/png;base64,{img.strip()}" />'
    if "text/markdown" in data:
        val = data["text/markdown"]
        if isinstance(val, list):
            val = "".join(val)
        return val
    if "text/plain" in data:
        val = data["text/plain"]
        if isinstance(val, list):
            val = "".join(val)
        return f"```\n{_trim(val)}\n```"
    if otype == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        return f"```\n{_trim(text)}\n```"
    return ""


def export_notebook(path: Path, out_path: Path, *, execute: bool, model: str) -> None:
    import nbformat

    nb = nbformat.read(str(path), as_version=4)

    if execute:
        try:
            from nbclient import NotebookClient
        except ImportError:
            print("error: --execute requires nbclient (pip install nbclient)", file=sys.stderr)
            sys.exit(2)
        client_exec = NotebookClient(nb, timeout=120, kernel_name="python3")
        client_exec.execute()

    client = _client()

    lines: list[str] = [f"# {path.stem}", ""]
    narrations: list[str] = []

    for i, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            if cell.cell_type == "markdown":
                lines.append(cell.source)
                lines.append("")
            continue

        outputs = cell.get("outputs") or []
        if not cell.source.strip() and not outputs:
            continue

        lines.append(f"## Cell {i}")
        lines.append("")
        if cell.source.strip():
            lines.append("```python")
            lines.append(cell.source.rstrip())
            lines.append("```")
            lines.append("")

        for output in outputs:
            rendered = _render_output_md(output)
            if rendered:
                lines.append("**Output:**")
                lines.append("")
                lines.append(rendered)
                lines.append("")

            payload = _output_to_payload(output)
            if payload is None:
                continue
            try:
                narration = _narrate(client, payload, model=model)
            except Exception as exc:
                narration = f"_(narration failed: {exc})_"
            if narration:
                narrations.append(f"Cell {i}: {narration}")
                lines.append("**Narration:**")
                lines.append("")
                lines.append(narration)
                lines.append("")

    if narrations:
        try:
            caveats = _caveats(client, narrations, model=model)
            lines.append("## Caveats")
            lines.append("")
            lines.append(caveats)
            lines.append("")
        except Exception as exc:
            lines.append(f"_(caveats generation failed: {exc})_")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sabre", description="Narrate Jupyter notebooks with Claude.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export a narrated markdown report of a notebook.")
    p_export.add_argument("notebook", type=Path, help="Path to the .ipynb file.")
    p_export.add_argument("--out", type=Path, default=None, help="Output path (defaults to <notebook>.report.md).")
    p_export.add_argument("--execute", action="store_true", help="Re-execute the notebook before narrating.")
    p_export.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model to use.")

    args = parser.parse_args(argv)

    if args.cmd == "export":
        nb_path = args.notebook
        if not nb_path.exists():
            print(f"error: {nb_path} not found", file=sys.stderr)
            return 2
        out_path = args.out or nb_path.with_suffix(".report.md")
        export_notebook(nb_path, out_path, execute=args.execute, model=args.model)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
