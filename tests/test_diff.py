import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sabre.diff import serialize_diff


def test_dataframe_diff_shape_and_added_columns():
    a = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    b = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
    p = serialize_diff(a, b)
    assert p.kind == "diff_dataframe"
    text = p.content[0]["text"]
    assert "Shape A: (3, 2)" in text
    assert "Shape B: (3, 3)" in text
    assert "Added columns: ['z']" in text


def test_dataframe_diff_numeric_shift_ranked():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"stable": rng.normal(0, 1, 100), "shifted": rng.normal(0, 1, 100)})
    b = a.copy()
    b["shifted"] = b["shifted"] + 10  # big shift
    p = serialize_diff(a, b)
    text = p.content[0]["text"]
    assert "Numeric shifts" in text
    # "shifted" should appear before "stable" in the ranked output
    assert text.index("shifted") < text.index("stable")


def test_dataframe_diff_null_deltas():
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"x": [1.0, None, None]})
    p = serialize_diff(a, b)
    text = p.content[0]["text"]
    assert "Null count changes" in text
    assert "0 -> 2" in text


def test_dataframe_diff_changed_row_sample():
    a = pd.DataFrame({"x": [1, 2, 3, 4]}, index=["a", "b", "c", "d"])
    b = pd.DataFrame({"x": [1, 99, 3, 4]}, index=["a", "b", "c", "d"])
    p = serialize_diff(a, b)
    text = p.content[0]["text"]
    assert "Rows with any value changed: 1 / 4" in text
    # pandas 3 wraps ints as np.int64(...); match either form
    assert ("x: 2 -> 99" in text) or ("-> np.int64(99)" in text)


def test_series_diff():
    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([10, 20, 30, 40, 50])
    p = serialize_diff(a, b)
    assert p.kind == "diff_series"
    assert "Describe A" in p.content[0]["text"]
    assert "Describe B" in p.content[0]["text"]


def test_ndarray_diff_same_shape_reports_abs_delta():
    a = np.zeros((4, 4))
    b = np.ones((4, 4)) * 2
    p = serialize_diff(a, b)
    assert p.kind == "diff_ndarray"
    text = p.content[0]["text"]
    assert "|A - B|" in text
    assert "mean=2" in text


def test_figure_diff_sends_two_images():
    fig_a, ax_a = plt.subplots()
    ax_a.plot([1, 2, 3])
    fig_b, ax_b = plt.subplots()
    ax_b.plot([3, 2, 1])
    p = serialize_diff(fig_a, fig_b)
    assert p.kind == "diff_figure"
    image_blocks = [c for c in p.content if c.get("type") == "image"]
    assert len(image_blocks) == 2
    plt.close(fig_a)
    plt.close(fig_b)


def test_exception_diff():
    try:
        1 / 0
    except ZeroDivisionError as e:
        err_a = e
    try:
        [][0]
    except IndexError as e:
        err_b = e
    p = serialize_diff(err_a, err_b)
    assert p.kind == "diff_exception"
    assert "ZeroDivisionError" in p.content[0]["text"]
    assert "IndexError" in p.content[0]["text"]


def test_generic_diff_fallback():
    p = serialize_diff({"a": 1}, {"a": 2})
    assert p.kind == "diff_generic"
    assert "dict" in p.content[0]["text"]
