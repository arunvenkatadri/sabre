from __future__ import annotations

import base64
import io
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass
class Payload:
    kind: str
    content: list  # Anthropic content blocks


def _df_to_text(df) -> str:
    parts = [
        f"Shape: {df.shape}",
        "",
        "Dtypes:",
        df.dtypes.to_string(),
        "",
        "Describe (numeric):",
        df.describe(include="number").to_string() if df.select_dtypes("number").shape[1] else "(no numeric columns)",
    ]
    obj_cols = df.select_dtypes(include=["object", "category"])
    if obj_cols.shape[1]:
        parts += ["", "Describe (non-numeric):", obj_cols.describe().to_string()]
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        parts += ["", "Null counts:", nulls.to_string()]
    parts += ["", f"Head ({min(10, len(df))} rows):", df.head(10).to_string()]
    return "\n".join(parts)


def _series_to_text(s) -> str:
    parts = [
        f"Length: {len(s)}  Dtype: {s.dtype}  Name: {s.name!r}",
        "",
        "Describe:",
        s.describe().to_string(),
    ]
    if s.dtype == "object" or str(s.dtype) == "category":
        vc = s.value_counts().head(20)
        parts += ["", f"Top {len(vc)} values:", vc.to_string()]
    nulls = s.isna().sum()
    if nulls:
        parts += ["", f"Nulls: {nulls}"]
    parts += ["", f"Head ({min(10, len(s))}):", s.head(10).to_string()]
    return "\n".join(parts)


def _ndarray_to_text(a) -> str:
    import numpy as np
    parts = [f"Shape: {a.shape}  Dtype: {a.dtype}  Size: {a.size}"]
    if a.size and np.issubdtype(a.dtype, np.number):
        parts += [
            f"min={np.nanmin(a):.6g}  max={np.nanmax(a):.6g}  "
            f"mean={np.nanmean(a):.6g}  std={np.nanstd(a):.6g}",
            f"nan count: {int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0}",
        ]
    sample = a if a.size <= 64 else a.ravel()[:64]
    parts += ["", "Sample (first 64 flat):", repr(sample)]
    return "\n".join(parts)


def _figure_to_png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _exception_to_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def serialize(obj: Any) -> Payload:
    # pandas
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return Payload("dataframe", [{"type": "text", "text": _df_to_text(obj)}])
        if isinstance(obj, pd.Series):
            return Payload("series", [{"type": "text", "text": _series_to_text(obj)}])
    except ImportError:
        pass

    # matplotlib — check before ndarray since Figure isn't an ndarray
    try:
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        fig = None
        if isinstance(obj, Figure):
            fig = obj
        elif isinstance(obj, Axes):
            fig = obj.figure
        if fig is not None:
            b64 = _figure_to_png_b64(fig)
            return Payload("figure", [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64,
                }},
                {"type": "text", "text": "Explain this plot."},
            ])
    except ImportError:
        pass

    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return Payload("ndarray", [{"type": "text", "text": _ndarray_to_text(obj)}])
    except ImportError:
        pass

    # exceptions
    if isinstance(obj, BaseException):
        return Payload("exception", [{"type": "text", "text": _exception_to_text(obj)}])

    # strings
    if isinstance(obj, str):
        text = obj if len(obj) <= 20000 else obj[:20000] + "\n... [truncated]"
        return Payload("text", [{"type": "text", "text": text}])

    # fallback
    r = repr(obj)
    if len(r) > 20000:
        r = r[:20000] + "\n... [truncated]"
    return Payload("generic", [{"type": "text", "text": f"type: {type(obj).__name__}\n\n{r}"}])
