from sabre.prompts import DATAFRAME, DIFF, SUGGEST_SUFFIX, system_for


def test_system_for_known_kinds():
    assert system_for("dataframe") == DATAFRAME


def test_system_for_diff_kinds_returns_diff_prompt():
    for kind in ("diff_dataframe", "diff_series", "diff_ndarray", "diff_figure", "diff_exception"):
        assert system_for(kind) == DIFF


def test_suggest_suffix_is_nonempty():
    assert "sabre-suggestions" in SUGGEST_SUFFIX
