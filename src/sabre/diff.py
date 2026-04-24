"""Compare two outputs and narrate what changed.

Dispatches on the pair's types. For each pair, produces a compact text
summary of what differs so Claude can focus on meaningful deltas rather
than re-describing both sides.
"""
from __future__ import annotations

from typing import Any

from .serialize import Payload, _figure_to_png_b64


def _df_diff_text(a, b) -> str:
    import numpy as np

    parts = [f"Shape A: {a.shape}   Shape B: {b.shape}"]

    cols_a, cols_b = set(a.columns), set(b.columns)
    added = sorted(cols_b - cols_a)
    dropped = sorted(cols_a - cols_b)
    if added:
        parts.append(f"Added columns: {added}")
    if dropped:
        parts.append(f"Dropped columns: {dropped}")

    shared = [c for c in a.columns if c in cols_b]

    dtype_changes = []
    for c in shared:
        if a[c].dtype != b[c].dtype:
            dtype_changes.append(f"  {c}: {a[c].dtype} -> {b[c].dtype}")
    if dtype_changes:
        parts += ["", "Dtype changes:"] + dtype_changes

    null_deltas = []
    for c in shared:
        na_a = int(a[c].isna().sum())
        na_b = int(b[c].isna().sum())
        if na_a != na_b:
            null_deltas.append(f"  {c}: nulls {na_a} -> {na_b}")
    if null_deltas:
        parts += ["", "Null count changes:"] + null_deltas

    numeric_shared = [
        c for c in shared
        if np.issubdtype(a[c].dtype, np.number) and np.issubdtype(b[c].dtype, np.number)
    ]
    if numeric_shared:
        stat_rows = []
        for c in numeric_shared:
            sa, sb = a[c], b[c]
            if len(sa) == 0 or len(sb) == 0:
                continue
            ma, mb = float(sa.mean()), float(sb.mean())
            sda, sdb = float(sa.std()), float(sb.std())
            denom = sda if sda else 1.0
            z = (mb - ma) / denom
            stat_rows.append((abs(z), c, ma, mb, sda, sdb))
        stat_rows.sort(reverse=True)
        top = stat_rows[:10]
        if top:
            parts += ["", "Numeric shifts (top 10, sorted by |Δmean / σ_A|):"]
            parts.append("  column            mean_A        mean_B        std_A        std_B")
            for _, c, ma, mb, sda, sdb in top:
                parts.append(f"  {c:<16}  {ma:<12.6g}  {mb:<12.6g}  {sda:<10.6g}  {sdb:<10.6g}")

    if a.shape == b.shape and list(a.columns) == list(b.columns):
        try:
            same_index = a.index.equals(b.index)
        except Exception:
            same_index = False
        if same_index:
            diff_mask = (a != b) & ~(a.isna() & b.isna())
            changed_rows = diff_mask.any(axis=1)
            n_changed = int(changed_rows.sum())
            parts += ["", f"Rows with any value changed: {n_changed} / {len(a)}"]
            if 0 < n_changed <= 50:
                idx = changed_rows[changed_rows].index[:10]
                parts += [f"Sample of changed rows (first {len(idx)}):"]
                for i in idx:
                    diffs = []
                    for c in a.columns:
                        if diff_mask.loc[i, c]:
                            diffs.append(f"{c}: {a.loc[i, c]!r} -> {b.loc[i, c]!r}")
                    parts.append(f"  [{i!r}] " + "; ".join(diffs[:5]))

    return "\n".join(parts)


def _series_diff_text(a, b) -> str:
    parts = [
        f"Length A: {len(a)}  Length B: {len(b)}",
        f"Dtype A: {a.dtype}  Dtype B: {b.dtype}",
    ]
    try:
        parts += ["", "Describe A:", a.describe().to_string(), "", "Describe B:", b.describe().to_string()]
    except Exception:
        pass
    if str(a.dtype) in ("object", "category") or str(b.dtype) in ("object", "category"):
        va = a.value_counts().head(10)
        vb = b.value_counts().head(10)
        parts += ["", "Top values A:", va.to_string(), "", "Top values B:", vb.to_string()]
    return "\n".join(parts)


def _ndarray_diff_text(a, b) -> str:
    import numpy as np

    parts = [
        f"Shape A: {a.shape}  Dtype A: {a.dtype}",
        f"Shape B: {b.shape}  Dtype B: {b.dtype}",
    ]
    if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        d = np.abs(a.astype(float) - b.astype(float))
        parts += [
            "",
            f"|A - B|: min={np.nanmin(d):.6g}  max={np.nanmax(d):.6g}  "
            f"mean={np.nanmean(d):.6g}",
            f"Exactly equal elements: {int((a == b).sum())} / {a.size}",
        ]
    else:
        if np.issubdtype(a.dtype, np.number):
            parts.append(f"A stats: mean={np.nanmean(a):.6g} std={np.nanstd(a):.6g}")
        if np.issubdtype(b.dtype, np.number):
            parts.append(f"B stats: mean={np.nanmean(b):.6g} std={np.nanstd(b):.6g}")
    return "\n".join(parts)


def _exception_diff_text(a: BaseException, b: BaseException) -> str:
    import traceback
    ta = "".join(traceback.format_exception(type(a), a, a.__traceback__))
    tb = "".join(traceback.format_exception(type(b), b, b.__traceback__))
    return f"Traceback A:\n{ta}\n---\nTraceback B:\n{tb}"


def serialize_diff(a: Any, b: Any) -> Payload:
    """Produce a Payload describing the delta between two objects."""
    try:
        import pandas as pd
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            return Payload("diff_dataframe", [{"type": "text", "text": _df_diff_text(a, b)}])
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return Payload("diff_series", [{"type": "text", "text": _series_diff_text(a, b)}])
    except ImportError:
        pass

    try:
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        def _fig(x):
            if isinstance(x, Figure):
                return x
            if isinstance(x, Axes):
                return x.figure
            return None

        fa, fb = _fig(a), _fig(b)
        if fa is not None and fb is not None:
            return Payload("diff_figure", [
                {"type": "text", "text": "Figure A:"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _figure_to_png_b64(fa),
                }},
                {"type": "text", "text": "Figure B:"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _figure_to_png_b64(fb),
                }},
                {"type": "text", "text": "What changed between A and B?"},
            ])
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return Payload("diff_ndarray", [{"type": "text", "text": _ndarray_diff_text(a, b)}])
    except ImportError:
        pass

    if isinstance(a, BaseException) and isinstance(b, BaseException):
        return Payload("diff_exception", [{"type": "text", "text": _exception_diff_text(a, b)}])

    ra, rb = repr(a), repr(b)
    if len(ra) > 10000:
        ra = ra[:10000] + "\n... [truncated]"
    if len(rb) > 10000:
        rb = rb[:10000] + "\n... [truncated]"
    text = f"A ({type(a).__name__}):\n{ra}\n\nB ({type(b).__name__}):\n{rb}"
    return Payload("diff_generic", [{"type": "text", "text": text}])
