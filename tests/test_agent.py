import json

from sabre.agent import _check_safe, _parse_suggestions


def test_safe_snippet_passes():
    ok, reason = _check_safe("df['price'].describe()")
    assert ok
    assert reason is None


def test_unsafe_import_blocked():
    ok, reason = _check_safe("import os\nos.listdir('.')")
    assert not ok
    assert "import" in reason


def test_unsafe_open_blocked():
    ok, reason = _check_safe("with open('/etc/passwd') as f: pass")
    assert not ok


def test_unsafe_network_blocked():
    ok, reason = _check_safe("requests.get('http://example.com')")
    assert not ok


def test_unsafe_shell_escape_blocked():
    ok, reason = _check_safe("!rm -rf /tmp/x")
    assert not ok


def test_unsafe_magic_blocked():
    ok, reason = _check_safe("%time expensive()")
    assert not ok


def test_parse_suggestions_extracts_fenced_json():
    suggestions = [
        {"title": "Inspect nulls", "code": "df.isna().sum()", "rationale": "See null pattern"},
        {"title": "Unsafe", "code": "import os", "rationale": "bad"},
    ]
    text = (
        "This is the narration.\n\n"
        "```sabre-suggestions\n"
        + json.dumps(suggestions)
        + "\n```\n"
    )
    narration, parsed = _parse_suggestions(text)
    assert narration == "This is the narration."
    assert len(parsed) == 2
    assert parsed[0].safe is True
    assert parsed[1].safe is False
    assert parsed[1].unsafe_reason is not None


def test_parse_suggestions_no_fence_returns_full_text():
    narration, parsed = _parse_suggestions("Just narration, no suggestions.")
    assert narration == "Just narration, no suggestions."
    assert parsed == []


def test_parse_suggestions_malformed_json_yields_empty():
    text = "Narration.\n\n```sabre-suggestions\nnot valid json\n```\n"
    narration, parsed = _parse_suggestions(text)
    assert narration == "Narration."
    assert parsed == []


def test_parse_suggestions_caps_at_three():
    items = [{"title": f"t{i}", "code": f"print({i})", "rationale": "r"} for i in range(5)]
    text = "n\n```sabre-suggestions\n" + json.dumps(items) + "\n```"
    _, parsed = _parse_suggestions(text)
    assert len(parsed) == 3
