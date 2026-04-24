"""Domain packs — swappable prompt addendums that specialize sabre's
narration for a specific kind of work (ML training, finance EDA, etc.).

A pack contributes two things:
- A `general` addendum applied to every output kind.
- Optional `per_kind` addendums for dataframe / figure / series / etc.

Both are appended to the base system prompt in `core._run` via
`system_suffix`. Users can switch packs with `sabre.use("ml-training")`
or register their own with `sabre.register_pack(Pack(...))`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pack:
    name: str
    description: str
    general: str = ""
    per_kind: dict[str, str] = field(default_factory=dict)

    def addendum_for(self, kind: str) -> str:
        parts: list[str] = []
        if self.general:
            parts.append(self.general)
        base_kind = kind[len("diff_"):] if kind.startswith("diff_") else kind
        per = self.per_kind.get(base_kind) or self.per_kind.get(kind)
        if per:
            parts.append(per)
        if not parts:
            return ""
        return "\n\nDomain focus (" + self.name + "): " + " ".join(parts)


ML_TRAINING = Pack(
    name="ml-training",
    description="Leakage, look-ahead bias, class imbalance, overfit signals.",
    general=(
        "The user is training or evaluating an ML model. Be alert to data "
        "leakage (target-in-features, identifiers that proxy for labels, "
        "group leakage across train/val/test), look-ahead bias in time "
        "splits, class or sample imbalance, and suspiciously high "
        "performance that usually indicates leakage. Flag near-zero "
        "validation performance as either a label bug or a truly hard task."
    ),
    per_kind={
        "dataframe": (
            "Call out columns that look like post-outcome labels, row "
            "identifiers that may proxy for groups, and class imbalance "
            "when a column looks like a target."
        ),
        "figure": (
            "For training curves: flag train/val divergence, plateaus, and "
            "non-monotonic behavior. For confusion matrices and ROC/PR "
            "curves: call out class-specific failure modes and "
            "chance-level performance."
        ),
        "series": (
            "If this looks like a prediction or probability series, flag "
            "degenerate distributions (all one class, near-constant "
            "scores) or suspicious calibration."
        ),
    },
)


FINANCE_EDA = Pack(
    name="finance-eda",
    description="Unit/currency mismatches, survivorship bias, look-ahead.",
    general=(
        "The user is doing financial exploratory analysis. Be alert to "
        "unit mismatches (prices vs. returns, basis points vs. percent, "
        "currencies), survivorship bias (surviving tickers over-represented), "
        "look-ahead bias (features using future information), and outliers "
        "that are often splits, corporate actions, or bad ticks rather "
        "than real moves."
    ),
    per_kind={
        "dataframe": (
            "Flag suspect date gaps, non-trading-day rows, and columns "
            "whose ranges suggest they are prices where you'd expect "
            "returns (or vice versa)."
        ),
        "figure": (
            "For price / return plots, call out structural breaks, "
            "regime changes, and suspicious spikes that may be data "
            "errors. For distribution plots, comment on fat tails and "
            "skew relative to a normal reference."
        ),
    },
)


BIO_STATS = Pack(
    name="bio-stats",
    description="Multiple comparisons, batch effects, effect size vs. p-value.",
    general=(
        "The user is analyzing biological or biomedical data. Be alert to "
        "multiple-comparisons issues (many p-values without correction), "
        "batch effects masquerading as biological signal, small effect "
        "sizes reported with low p-values due to large N, and "
        "technical replicates being treated as biological replicates."
    ),
    per_kind={
        "dataframe": (
            "If columns include p-values, flag whether any correction "
            "appears to have been applied. Call out potential batch "
            "variables (plate, run, date, site) that should be examined "
            "as covariates."
        ),
        "figure": (
            "For volcano plots, heatmaps, and PCA plots: call out "
            "clustering that follows technical variables rather than "
            "biological conditions."
        ),
    },
)


DATA_QUALITY = Pack(
    name="data-quality",
    description="Nulls, duplicates, outliers, encoding issues, distribution shifts.",
    general=(
        "The user is doing data quality / validation work. Be strict. "
        "Flag missingness patterns (MCAR vs. structured), duplicates "
        "(exact vs. near), outliers that look like unit errors or "
        "sentinel values (999, -1, 1970-01-01), dtype mismatches, and "
        "encoding issues (mixed cases, stray whitespace, mojibake)."
    ),
    per_kind={
        "dataframe": (
            "Report the single most suspicious column and why. Call out "
            "sentinel-like values in numeric columns and high-cardinality "
            "near-duplicate categories in string columns."
        ),
    },
)


_BUILTIN: dict[str, Pack] = {
    p.name: p for p in (ML_TRAINING, FINANCE_EDA, BIO_STATS, DATA_QUALITY)
}
_REGISTRY: dict[str, Pack] = dict(_BUILTIN)
_ACTIVE: Pack | None = None


def use(name: str | None) -> Pack | None:
    """Activate a pack by name, or pass None to clear. Returns the active pack."""
    global _ACTIVE
    if name is None:
        _ACTIVE = None
        return None
    pack = _REGISTRY.get(name)
    if pack is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown pack: {name!r}. Known: {known}")
    _ACTIVE = pack
    return pack


def register_pack(pack: Pack) -> None:
    """Add or replace a pack in the registry. Useful for domain teams."""
    if not isinstance(pack, Pack):
        raise TypeError("register_pack expects a Pack instance")
    _REGISTRY[pack.name] = pack


def current_pack() -> Pack | None:
    return _ACTIVE


def list_packs() -> list[tuple[str, str]]:
    return [(p.name, p.description) for p in _REGISTRY.values()]


def pack_addendum_for(kind: str) -> str:
    """Return the system-prompt suffix contributed by the active pack, if any."""
    if _ACTIVE is None:
        return ""
    return _ACTIVE.addendum_for(kind)


def _reset_for_tests() -> None:
    """Test helper: clear user-registered packs and active selection."""
    global _REGISTRY, _ACTIVE
    _REGISTRY = dict(_BUILTIN)
    _ACTIVE = None
