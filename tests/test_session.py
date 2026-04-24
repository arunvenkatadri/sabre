import numpy as np
import pandas as pd

from sabre import session
from sabre.session import Session, _summarize


def test_summarize_dataframe():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    kind, summary = _summarize(df)
    assert kind == "dataframe"
    assert "shape=(3, 2)" in summary
    assert "cols_with_nulls=1" in summary


def test_summarize_series():
    s = pd.Series([1, 2, 3], name="foo")
    kind, summary = _summarize(s)
    assert kind == "series"
    assert "'foo'" in summary
    assert "len=3" in summary


def test_summarize_ndarray_stats():
    a = np.array([1.0, 2.0, 3.0])
    kind, summary = _summarize(a)
    assert kind == "ndarray"
    assert "mean=2" in summary


def test_summarize_exception():
    try:
        1 / 0
    except ZeroDivisionError as e:
        kind, summary = _summarize(e)
    assert kind == "exception"
    assert "ZeroDivisionError" in summary


def test_summarize_uninteresting_returns_none():
    assert _summarize("hello") is None
    assert _summarize({"k": "v"}) is None
    assert _summarize(42) is None


def test_session_disabled_by_default():
    s = Session()
    s.add("dataframe", "x")
    assert s.entries == []
    assert s.context_text() is None


def test_session_enable_and_add():
    s = Session(enabled=True)
    s.add("dataframe", "df shape=(10, 3)")
    s.add("series", "series len=10")
    ctx = s.context_text()
    assert ctx is not None
    assert "Cell 1 [dataframe]" in ctx
    assert "Cell 2 [series]" in ctx


def test_session_compaction_keeps_window():
    from sabre.session import COMPACT_EVERY, WINDOW

    s = Session(enabled=True)
    for i in range(35):
        s.add("ndarray", f"entry {i}")
    # Compaction is lazy (every COMPACT_EVERY cells) to keep the prefix
    # cache-stable, so the bound is WINDOW + COMPACT_EVERY - 1.
    assert len(s.entries) <= WINDOW + COMPACT_EVERY - 1
    assert s.compacted != ""
    assert "entry 34" in s.context_text()


def test_session_reset():
    s = Session(enabled=True)
    s.add("dataframe", "x")
    s.reset()
    assert s.entries == []
    assert s.cell_counter == 0


def test_module_level_enable_disable_idempotent():
    session.enable_memory()
    session.enable_memory()  # should not raise or double-register
    session.disable_memory()
    session.disable_memory()  # should not raise on second call
