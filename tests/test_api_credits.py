"""Local credit observation must never become account or billing authority."""

import dataclasses
import json

import pytest

from claude_swap.api_credits import (
    CONFIG_NAME, CreditMonitor, grouped_accounts, request_cost, timestamp, valid_estimates,
)
from claude_swap.models import AccountSnapshot, AccountsSnapshot
from claude_swap.session import session_dir_for
from claude_swap.usage_store import UsageEntry

START = "2026-10-11T09:00:00Z"
NOW = timestamp("2026-10-11T10:00:00Z")


def account(number, email):
    return AccountSnapshot(str(number), email, "", "", False, "oauth", True, UsageEntry())


def setup(root):
    # Nonconsecutive IDs intentionally exercise grouping without renumbering.
    parent = account(2, "owner@example.com")
    api = account(9, "setup-token-9@token.local")
    snap = AccountsSnapshot(None, (parent, account(3, "other@example.com"), api), NOW)
    (root / CONFIG_NAME).write_text(json.dumps({"9": {
        "group_with": "2", "source_identity": [api.email, ""],
        "group_identity": [parent.email, ""], "monthly_credit": 200, "since": START,
    }}))
    directory = session_dir_for(root, api.number, api.email) / "projects" / "project"
    directory.mkdir(parents=True)
    return snap, directory / "session.jsonl"


def usage(output=1000):
    return {
        "input_tokens": 1000, "output_tokens": output, "cache_read_input_tokens": 10000,
        "cache_creation_input_tokens": 3000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000},
        "speed": "standard", "inference_geo": "global",
    }


def message(key="request-a", at="2026-10-11T09:30:00Z", output=1000, model="claude-opus-5-5"):
    return {"type": "assistant", "timestamp": at, "requestId": key,
            "message": {"id": "message-" + key, "model": model, "usage": usage(output)}}


def append(path, *records):
    with path.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def test_prices_include_distinct_cache_write_ttls():
    # $0.004 input + $0.020 output + $0.002 reads + $0.005 5m + $0.016 1h.
    assert request_cost("claude-opus-5-5", usage()) == pytest.approx(.047)
    assert request_cost("claude-opus-5-5", {**usage(), "inference_geo": "us"}) == pytest.approx(.0517)


@pytest.mark.parametrize("changes", [
    {"speed": "fast"}, {"service_tier": "batch"}, {"input_tokens": -1},
    {"cache_creation": {}}, {"server_tool_use": {"web_search_requests": 1}},
])
def test_unpriced_or_invalid_usage_stays_unknown(changes):
    assert request_cost("claude-opus-5-5", {**usage(), **changes}) is None
    assert request_cost("unknown-model", usage()) is None


def test_deduplicates_streaming_rows_and_copied_logs(tmp_path):
    snap, path = setup(tmp_path)
    append(path, message(output=100), message(output=1000))
    append(path.with_name("copy.jsonl"), message())
    estimate = CreditMonitor(tmp_path).scan(snap, now=NOW)["9"]
    assert estimate.requests == 1
    assert estimate.spent == pytest.approx(.047)
    assert estimate.remaining == pytest.approx(199.953)
    assert estimate.last_used == timestamp("2026-10-11T09:30:00Z")
    assert not estimate.incomplete


def test_incremental_poll_waits_for_complete_line_then_handles_truncation(tmp_path):
    snap, path = setup(tmp_path)
    monitor = CreditMonitor(tmp_path)
    row = json.dumps(message())
    path.write_text(row[:20])
    assert monitor.scan(snap, now=NOW)["9"].requests == 0
    with path.open("a") as stream:
        stream.write(row[20:] + "\n")
    assert monitor.scan(snap, now=NOW)["9"].requests == 1
    offset = monitor.logs[path].offset
    assert monitor.scan(snap, now=NOW)["9"].spent == pytest.approx(.047)
    assert monitor.logs[path].offset == offset  # no reread of completed lines
    path.write_text("")
    assert monitor.scan(snap, now=NOW)["9"].requests == 0
    append(path, message("new"))
    assert monitor.scan(snap, now=NOW)["9"].requests == 1


def test_counts_only_tracking_period_and_does_not_guess_monthly_reset(tmp_path):
    snap, path = setup(tmp_path)
    append(path, message("old", at="2026-10-10T09:00:00Z"), message(),
           message("future", at="2026-11-11T09:00:00Z"))
    monitor = CreditMonitor(tmp_path)
    assert monitor.scan(snap, now=NOW)["9"].requests == 1
    # Next calendar month is not evidence that the billing cycle has renewed.
    assert monitor.scan(snap, now=timestamp("2026-11-12"))["9"].requests == 2


def test_exhaustion_is_observed_error_not_estimated_budget(tmp_path):
    snap, path = setup(tmp_path)
    quoted = {"type": "user", "timestamp": START,
              "message": {"content": "credit balance is too low"}}
    append(path, quoted, message())
    monitor = CreditMonitor(tmp_path)
    assert not monitor.scan(snap, now=NOW)["9"].exhausted
    error = {"type": "assistant", "timestamp": "2026-10-11T09:40:00Z",
             "isApiErrorMessage": True, "message": {"model": "<synthetic>",
             "content": [{"type": "text", "text": "Your credit balance is too low to access the Anthropic API."}]}}
    append(path, error)
    assert monitor.scan(snap, now=NOW)["9"].exhausted
    append(path, message("recovered", at="2026-10-11T09:50:00Z"))
    assert not monitor.scan(snap, now=NOW)["9"].exhausted


def test_unknown_price_or_unreadable_log_does_not_invent_remaining(tmp_path, monkeypatch):
    snap, path = setup(tmp_path)
    append(path, message(model="new-model"))
    estimate = CreditMonitor(tmp_path).scan(snap, now=NOW)["9"]
    assert estimate.requests == 1 and estimate.incomplete
    assert estimate.remaining is None
    original = type(path).open

    def deny(self, *args, **kwargs):
        if self == path:
            raise PermissionError("test-owned denied log")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", deny)
    assert CreditMonitor(tmp_path).scan(snap, now=NOW)["9"].remaining is None


def test_slot_replacement_invalidates_display_metadata(tmp_path):
    snap, path = setup(tmp_path)
    append(path, message())
    changed = dataclasses.replace(snap, accounts=(snap.accounts[0], account(9, "different@example.com")))
    assert CreditMonitor(tmp_path).scan(changed, now=NOW) == {}
    changed_parent = dataclasses.replace(snap, accounts=(account(2, "different@example.com"), snap.accounts[2]))
    assert CreditMonitor(tmp_path).scan(changed_parent, now=NOW) == {}
    previous = CreditMonitor(tmp_path).scan(snap, now=NOW)
    assert valid_estimates(changed, previous) == {}
    assert [a.number for a in grouped_accounts(changed, previous).accounts] == ["2", "9"]


def test_grouping_preserves_original_accounts_and_each_switch_target(tmp_path):
    snap, path = setup(tmp_path)
    estimates = CreditMonitor(tmp_path).scan(snap, now=NOW)
    grouped = grouped_accounts(snap, estimates)
    assert [a.number for a in grouped.accounts] == ["2", "9", "3"]
    assert {a.number for a in grouped.accounts} == {a.number for a in snap.accounts}
    assert snap.accounts[1].number == "3"
    assert all(a.kind == "oauth" for a in grouped.accounts)


def test_display_labels_estimates_and_incomplete_or_exhausted_status(tmp_path):
    from claude_swap.tui.widgets import credit_card_text

    snap, path = setup(tmp_path)
    append(path, message())
    estimate = CreditMonitor(tmp_path).scan(snap, now=NOW)["9"]
    text = credit_card_text(estimate, now=NOW).plain
    assert "$200.00/mo" in text and "~$0.05 used" in text and "~$199.95 left" in text
    assert "owner@example.com" in text and "1 API requests" in text
    assert "last used 30m ago" in text and "since 2026-10-11" in text
    assert "remaining unknown" in credit_card_text(dataclasses.replace(estimate, incomplete=True), now=NOW).plain
    assert "credit exhausted (API error)" in credit_card_text(
        dataclasses.replace(estimate, credit_error_at=NOW), now=NOW).plain


def test_cli_configuration_changes_only_observer_file(tmp_path, monkeypatch, capsys):
    from claude_swap import cli

    snap, path = setup(tmp_path)
    (tmp_path / CONFIG_NAME).unlink()
    roster = tmp_path / "sequence.json"
    roster.write_text("unchanged roster")
    credential = tmp_path / "credentials.fake"
    credential.write_text("unchanged credential")

    class Switcher:
        backup_dir = tmp_path

        def accounts_snapshot(self, fetch):
            assert fetch == set()
            return snap

        def resolve_account(self, number):
            acc = next(a for a in snap.accounts if a.number == number)
            return acc.number, acc.email, acc.org_uuid

        def _get_sequence_data(self):
            return {"accounts": {"9": {"added": START}}}

        def _write_json(self, path, data):
            path.write_text(json.dumps(data))

    monkeypatch.setattr(cli, "ClaudeAccountSwitcher", Switcher)
    monkeypatch.setattr("time.time", lambda: NOW)
    cli._credits_command(["set", "9", "--group-with", "2", "--monthly-credit", "200"])
    config = json.loads((tmp_path / CONFIG_NAME).read_text())
    assert config["9"]["since"] == START and config["9"]["source_identity"] == [snap.accounts[2].email, ""]
    assert roster.read_text() == "unchanged roster" and credential.read_text() == "unchanged credential"
    append(path, message())
    capsys.readouterr()
    cli._credits_command(["--json"])
    assert json.loads(capsys.readouterr().out)["9"]["requests"] == 1
    cli._credits_command(["unset", "9"])
    assert json.loads((tmp_path / CONFIG_NAME).read_text()) == {}
