"""Rolling notebook-level context for sabre.

A per-process singleton collects compact summaries of cell outputs as the
user runs cells. When `explain()` is called, the summary is injected as
an extra cached system block so Claude can reason across the notebook
(e.g. "this metric was 0.82 four cells ago, now 0.61").

Design notes:
- Only `interesting` output types contribute entries (DataFrames, Series,
  ndarrays, Figures, Exceptions). Figures are skipped in the rolling
  summary because re-sending images would blow the budget.
- A fixed window (last N entries) is kept verbatim. Older entries are
  compacted into a single bucket that regenerates every `COMPACT_EVERY`
  cells, so the prefix stays cache-stable between compactions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WINDOW = 20
COMPACT_EVERY = 10
SUMMARY_CHAR_LIMIT = 400


@dataclass
class Entry:
    cell: int
    kind: str
    summary: str


@dataclass
class Session:
    entries: list[Entry] = field(default_factory=list)
    compacted: str = ""
    cells_since_compaction: int = 0
    enabled: bool = False
    cell_counter: int = 0

    def add(self, kind: str, summary: str) -> None:
        if not self.enabled:
            return
        self.cell_counter += 1
        self.entries.append(Entry(self.cell_counter, kind, _trim(summary)))
        self.cells_since_compaction += 1
        if len(self.entries) > WINDOW and self.cells_since_compaction >= COMPACT_EVERY:
            self._compact()

    def _compact(self) -> None:
        overflow = self.entries[:-WINDOW]
        if not overflow:
            return
        lines = [f"Cell {e.cell} [{e.kind}]: {e.summary}" for e in overflow]
        if self.compacted:
            lines.insert(0, self.compacted)
        joined = "\n".join(lines)
        if len(joined) > 4000:
            joined = joined[-4000:]
        self.compacted = joined
        self.entries = self.entries[-WINDOW:]
        self.cells_since_compaction = 0

    def context_text(self) -> str | None:
        if not self.enabled or (not self.entries and not self.compacted):
            return None
        parts = ["Prior context in this notebook (most recent last):"]
        if self.compacted:
            parts += ["", "Earlier (compacted):", self.compacted]
        if self.entries:
            parts += ["", "Recent:"]
            parts += [f"Cell {e.cell} [{e.kind}]: {e.summary}" for e in self.entries]
        return "\n".join(parts)

    def recap(self) -> str:
        if not self.entries and not self.compacted:
            return "(no entries yet — sabre memory may be disabled; call sabre.enable_memory())"
        return self.context_text() or ""

    def reset(self) -> None:
        self.entries.clear()
        self.compacted = ""
        self.cells_since_compaction = 0
        self.cell_counter = 0


_SESSION = Session()


def _trim(text: str, limit: int = SUMMARY_CHAR_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _summarize(obj: Any) -> tuple[str, str] | None:
    """Return (kind, short_summary) for interesting objects, else None."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            null_cols = int((obj.isna().sum() > 0).sum())
            return (
                "dataframe",
                f"DataFrame shape={obj.shape}, cols={list(obj.columns)[:8]}"
                f"{'...' if obj.shape[1] > 8 else ''}, cols_with_nulls={null_cols}",
            )
        if isinstance(obj, pd.Series):
            return (
                "series",
                f"Series name={obj.name!r}, len={len(obj)}, dtype={obj.dtype}",
            )
    except ImportError:
        pass

    try:
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        if isinstance(obj, (Figure, Axes)):
            return ("figure", "(matplotlib figure — not captured in memory to save budget)")
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            summary = f"ndarray shape={obj.shape}, dtype={obj.dtype}"
            if obj.size and np.issubdtype(obj.dtype, np.number):
                summary += f", mean={np.nanmean(obj):.4g}, std={np.nanstd(obj):.4g}"
            return ("ndarray", summary)
    except ImportError:
        pass

    if isinstance(obj, BaseException):
        return ("exception", f"{type(obj).__name__}: {obj}")

    return None


def _post_run_cell(result) -> None:
    """IPython post_run_cell hook: if a cell produced an interesting value,
    log a short summary into the rolling session."""
    if not _SESSION.enabled:
        return
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        out = ip.user_ns.get("_")
        if out is None:
            return
        summary = _summarize(out)
        if summary is None:
            return
        kind, text = summary
        _SESSION.add(kind, text)
    except Exception:
        # Memory is best-effort; never let it break a cell.
        return


def enable_memory() -> None:
    """Turn on rolling notebook memory for this session."""
    _SESSION.enabled = True
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and _post_run_cell not in ip.events.callbacks.get("post_run_cell", []):
            ip.events.register("post_run_cell", _post_run_cell)
    except ImportError:
        pass


def disable_memory() -> None:
    _SESSION.enabled = False
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            try:
                ip.events.unregister("post_run_cell", _post_run_cell)
            except ValueError:
                pass
    except ImportError:
        pass


def current_context_block() -> str | None:
    """Used by core.explain — returns the context string or None."""
    return _SESSION.context_text()


def recap() -> str:
    return _SESSION.recap()


def reset_memory() -> None:
    _SESSION.reset()
