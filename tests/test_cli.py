"""Tests for the sabre export CLI — focus on the pieces that don't need
live API calls: output payload conversion and markdown rendering.
"""
import base64

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sabre.cli import _output_to_payload, _render_output_md


def _png_b64() -> str:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_output_to_payload_figure():
    output = {
        "output_type": "display_data",
        "data": {"image/png": _png_b64(), "text/plain": "<Figure>"},
    }
    p = _output_to_payload(output)
    assert p is not None
    assert p.kind == "figure"
    assert p.content[0]["type"] == "image"


def test_output_to_payload_text_plain():
    output = {
        "output_type": "execute_result",
        "data": {"text/plain": "42"},
    }
    p = _output_to_payload(output)
    assert p is not None
    assert p.kind == "text"
    assert p.content[0]["text"] == "42"


def test_output_to_payload_error():
    output = {
        "output_type": "error",
        "ename": "ZeroDivisionError",
        "evalue": "division by zero",
        "traceback": ["Traceback (most recent call last):", "ZeroDivisionError: division by zero"],
    }
    p = _output_to_payload(output)
    assert p is not None
    assert p.kind == "exception"
    assert "ZeroDivisionError" in p.content[0]["text"]


def test_output_to_payload_stream():
    output = {"output_type": "stream", "name": "stdout", "text": "hello\n"}
    p = _output_to_payload(output)
    assert p is not None
    assert p.kind == "text"
    assert "hello" in p.content[0]["text"]


def test_output_to_payload_empty_returns_none():
    output = {"output_type": "display_data", "data": {}}
    assert _output_to_payload(output) is None


def test_render_output_md_text():
    output = {"output_type": "execute_result", "data": {"text/plain": "42"}}
    md = _render_output_md(output)
    assert "```" in md
    assert "42" in md


def test_render_output_md_image_embeds_data_url():
    b64 = _png_b64()
    output = {"output_type": "display_data", "data": {"image/png": b64}}
    md = _render_output_md(output)
    assert "data:image/png;base64," in md


def test_render_output_md_list_text_joined():
    output = {
        "output_type": "execute_result",
        "data": {"text/plain": ["line1\n", "line2"]},
    }
    md = _render_output_md(output)
    assert "line1" in md
    assert "line2" in md
