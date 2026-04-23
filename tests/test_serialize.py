import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sabre.serialize import serialize


def test_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = serialize(df)
    assert p.kind == "dataframe"
    assert p.content[0]["type"] == "text"
    assert "Shape: (3, 2)" in p.content[0]["text"]


def test_series():
    s = pd.Series([1, 2, 3], name="nums")
    p = serialize(s)
    assert p.kind == "series"
    assert "nums" in p.content[0]["text"]


def test_ndarray():
    a = np.arange(12).reshape(3, 4)
    p = serialize(a)
    assert p.kind == "ndarray"
    assert "Shape: (3, 4)" in p.content[0]["text"]


def test_figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    p = serialize(fig)
    assert p.kind == "figure"
    assert p.content[0]["type"] == "image"
    assert p.content[0]["source"]["media_type"] == "image/png"
    plt.close(fig)


def test_axes_uses_parent_figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    p = serialize(ax)
    assert p.kind == "figure"
    plt.close(fig)


def test_exception():
    try:
        1 / 0
    except ZeroDivisionError as e:
        p = serialize(e)
    assert p.kind == "exception"
    assert "ZeroDivisionError" in p.content[0]["text"]


def test_string():
    p = serialize("hello world")
    assert p.kind == "text"
    assert p.content[0]["text"] == "hello world"


def test_string_truncation():
    big = "x" * 30000
    p = serialize(big)
    assert "[truncated]" in p.content[0]["text"]


def test_generic_fallback():
    p = serialize({"a": 1})
    assert p.kind == "generic"
    assert "dict" in p.content[0]["text"]
