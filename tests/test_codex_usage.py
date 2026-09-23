"""Codex read-only app-server integration and limit conversion."""

from claude_swap.codex_usage import fetch_codex_usage, parse_rate_limits


def test_parse_rate_limits_uses_provider_windows_and_reset_times():
    usage = parse_rate_limits({
        "planType": "pro",
        "rateLimits": {
            "limitId": "codex",
            "primary": {"usedPercent": 13, "windowDurationMins": 300, "resetsAt": 2000000000},
            "secondary": {"usedPercent": 87, "windowDurationMins": 10080, "resetsAt": 2000000100},
        },
        "rateLimitsByLimitId": {
            "codex": {"limitId": "codex"},
            "spark": {"limitName": "Spark", "primary": {
                "usedPercent": 42, "windowDurationMins": 300, "resetsAt": None,
            }},
        },
    })
    assert usage.plan == "pro"
    assert [(w.label, w.pct) for w in usage.windows] == [
        ("5h", 13), ("7d", 87), ("Spark 5h", 42),
    ]
    assert usage.windows[0].resets_at == "2033-05-18T03:33:20+00:00"
    assert usage.windows[2].resets_at is None


def test_missing_limits_prompt_sign_in_without_fake_bars():
    usage = parse_rate_limits({"rateLimits": None})
    assert not usage.windows
    assert "codex login" in usage.status


def test_missing_cli_is_reported(monkeypatch):
    monkeypatch.setattr("claude_swap.codex_usage.shutil.which", lambda _name: None)
    assert fetch_codex_usage().status == "Codex CLI not found"
