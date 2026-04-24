import pytest

from sabre.packs import (
    Pack,
    _reset_for_tests,
    current_pack,
    list_packs,
    pack_addendum_for,
    register_pack,
    use,
)


@pytest.fixture(autouse=True)
def _clean():
    _reset_for_tests()
    yield
    _reset_for_tests()


def test_no_active_pack_yields_empty_addendum():
    assert current_pack() is None
    assert pack_addendum_for("dataframe") == ""


def test_use_activates_builtin_pack():
    pack = use("ml-training")
    assert pack is not None
    assert pack.name == "ml-training"
    assert current_pack() is pack


def test_use_none_clears_pack():
    use("ml-training")
    use(None)
    assert current_pack() is None
    assert pack_addendum_for("dataframe") == ""


def test_use_unknown_pack_raises():
    with pytest.raises(ValueError, match="Unknown pack"):
        use("nonexistent-pack")


def test_addendum_includes_general_and_per_kind():
    use("ml-training")
    addendum = pack_addendum_for("dataframe")
    assert "ml-training" in addendum
    # The general ML-training text mentions leakage.
    assert "leakage" in addendum.lower()
    # The dataframe-specific addition mentions labels/identifiers.
    assert "label" in addendum.lower() or "identifier" in addendum.lower()


def test_diff_kind_reuses_base_per_kind_addendum():
    use("data-quality")
    df_addendum = pack_addendum_for("dataframe")
    diff_addendum = pack_addendum_for("diff_dataframe")
    # The dataframe-specific language should appear in both.
    assert "suspicious" in df_addendum.lower()
    assert "suspicious" in diff_addendum.lower()


def test_kind_with_no_per_kind_falls_back_to_general_only():
    use("ml-training")
    # 'text' kind has no per_kind entry in ml-training.
    addendum = pack_addendum_for("text")
    assert "Domain focus" in addendum
    # The dataframe-specific string should NOT be present.
    assert "identifier" not in addendum.lower() or "leakage" in addendum.lower()


def test_register_custom_pack():
    custom = Pack(
        name="my-domain",
        description="custom",
        general="Be especially alert to FROBNICATION errors.",
        per_kind={"dataframe": "Look at the FOO column first."},
    )
    register_pack(custom)
    use("my-domain")
    addendum = pack_addendum_for("dataframe")
    assert "FROBNICATION" in addendum
    assert "FOO" in addendum


def test_register_pack_replaces_same_name():
    original = Pack(name="dup", description="v1", general="v1 text")
    updated = Pack(name="dup", description="v2", general="v2 text")
    register_pack(original)
    register_pack(updated)
    use("dup")
    assert "v2 text" in pack_addendum_for("dataframe")
    assert "v1 text" not in pack_addendum_for("dataframe")


def test_register_pack_type_check():
    with pytest.raises(TypeError):
        register_pack({"name": "x"})  # type: ignore[arg-type]


def test_list_packs_includes_builtins():
    names = {name for name, _ in list_packs()}
    assert {"ml-training", "finance-eda", "bio-stats", "data-quality"} <= names


def test_pack_is_frozen():
    pack = use("ml-training")
    with pytest.raises(Exception):
        pack.name = "changed"  # type: ignore[misc]
