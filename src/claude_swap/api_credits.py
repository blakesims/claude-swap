"""Read-only API credit estimates from account-local Claude transcripts.

Separate display metadata deliberately leaves account identities, credential kinds
and rotation untouched. An allowance is an assumed starting balance, not a billing
measurement. Optional monthly periods use the operator's claim date, not inferred
billing metadata.
"""

from __future__ import annotations

import calendar
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from claude_swap.models import AccountSnapshot, AccountsSnapshot
from claude_swap.session import session_dir_for

CONFIG_NAME = "credit-accounts.json"
# USD per million tokens: input, output, cache read, 5m write, 1h write.
# https://platform.claude.com/docs/en/about-claude/pricing
RATES = {
    "claude-opus-5-5": (4, 20, .20, 5, 8),
    "claude-opus-5": (5, 25, .50, 6.25, 10),
    "claude-sonnet-5-5": (2, 10, .10, 2.50, 4),
    "claude-sonnet-5": (2, 10, .20, 2.50, 4),
    "claude-sonnet-4-6": (3, 15, .30, 3.75, 6),
    "claude-haiku-4-5": (1, 5, .10, 1.25, 2),
}
CREDIT_ERRORS = ("credit balance is too low", "credit balance too low")


def timestamp(value: str) -> float:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def monthly_window(since: str, now: float) -> tuple[str, str]:
    """Calendar months from a supplied claim date, retaining month-end anchors."""
    anchor = datetime.fromtimestamp(timestamp(since), tz=timezone.utc)
    current = datetime.fromtimestamp(now, tz=timezone.utc)
    elapsed = max(0, (current.year - anchor.year) * 12 + current.month - anchor.month)

    def boundary(months: int) -> datetime:
        year, month = divmod(anchor.year * 12 + anchor.month - 1 + months, 12)
        month += 1
        return anchor.replace(year=year, month=month, day=min(anchor.day, calendar.monthrange(year, month)[1]))

    if elapsed and boundary(elapsed) > current:
        elapsed -= 1
    return boundary(elapsed).isoformat(), boundary(elapsed + 1).isoformat()


def identity(account: AccountSnapshot) -> list[str]:
    return [account.email, account.org_uuid]


@dataclass(frozen=True)
class CreditEstimate:
    number: str
    group_with: str
    email: str
    monthly_credit: float
    since: str
    spent: float = 0.0
    requests: int = 0
    last_used: float | None = None
    credit_error_at: float | None = None
    incomplete: bool = False
    source_identity: tuple[str, str] = ("", "")
    group_org_uuid: str = ""
    resets_at: str | None = None

    @property
    def pct(self) -> float | None:
        return None if self.incomplete else min(100.0, max(0.0, self.spent / self.monthly_credit * 100))

    @property
    def remaining(self) -> float | None:
        return None if self.incomplete else max(0.0, self.monthly_credit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.credit_error_at is not None and (
            self.last_used is None or self.credit_error_at >= self.last_used
        )


def load_credit_accounts(root: Path) -> dict:
    try:
        data = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError("credit account settings must be an object")
    for number, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"invalid credit settings for account {number}")
        amount = entry.get("monthly_credit")
        if not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
            raise ValueError(f"invalid monthly credit for account {number}")
        timestamp(entry["since"])
        if not isinstance(entry.get("monthly_cycle", False), bool):
            raise ValueError(f"invalid monthly cycle for account {number}")
        if not isinstance(entry.get("source_identity"), list) or not entry.get("group_with"):
            raise ValueError(f"missing account identity for account {number}")
    return data


def valid_estimates(snap: AccountsSnapshot, estimates: dict[str, CreditEstimate]) -> dict[str, CreditEstimate]:
    """A delayed scan must not decorate a replacement account in a newer snapshot."""
    accounts = {a.number: a for a in snap.accounts}
    return {
        number: estimate for number, estimate in estimates.items()
        if (acc := accounts.get(number)) is not None
        and tuple(identity(acc)) == estimate.source_identity
        and (parent := accounts.get(estimate.group_with)) is not None
        and (parent.email, parent.org_uuid) == (estimate.email, estimate.group_org_uuid)
    }


def grouped_accounts(snap: AccountsSnapshot, estimates: dict[str, CreditEstimate]) -> AccountsSnapshot:
    """Group the watch display only. Every slot remains individually selectable."""
    from dataclasses import replace

    accounts = {a.number: a for a in snap.accounts}
    estimates = valid_estimates(snap, estimates)
    children: dict[str, list[str]] = {}
    for number, estimate in estimates.items():
        if number in accounts and estimate.group_with in accounts and estimate.group_with not in estimates:
            children.setdefault(estimate.group_with, []).append(number)
    child_numbers = {n for ns in children.values() for n in ns}
    ordered = []
    for acc in snap.accounts:
        if acc.number in child_numbers:
            continue
        ordered.append(acc)
        ordered.extend(accounts[n] for n in children.get(acc.number, []))
    return replace(snap, accounts=tuple(ordered))


def request_cost(model: str, usage: dict) -> float | None:
    """Unknown prices/modifiers stay unknown, never silently count as free."""
    rates = RATES.get(model)
    if rates is None or usage.get("speed", "standard") not in (None, "standard"):
        return None
    if usage.get("service_tier", "standard") not in (None, "standard"):
        return None
    if usage.get("inference_geo", "global") not in (None, "global", "not_available", "us"):
        return None
    if any((usage.get("server_tool_use") or {}).values()):
        return None  # do not omit unpriced server-tool charges
    cache = usage.get("cache_creation") or {}
    write5 = cache.get("ephemeral_5m_input_tokens", 0)
    write1 = cache.get("ephemeral_1h_input_tokens", 0)
    amounts = (
        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        usage.get("cache_read_input_tokens", 0), write5, write1,
    )
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in amounts):
        return None
    total_write = usage.get("cache_creation_input_tokens", 0)
    if not isinstance(total_write, (int, float)) or total_write != write5 + write1:
        return None  # no invented TTL for cache writes
    cost = sum(n * rate for n, rate in zip(amounts, rates)) / 1_000_000
    return cost * (1.1 if usage.get("inference_geo") == "us" else 1)


@dataclass
class _Log:
    inode: int = 0
    offset: int = 0
    requests: dict = field(default_factory=dict)
    credit_error_at: float | None = None
    incomplete: bool = False

    def update(self, path: Path) -> None:
        """Read complete appended lines only. A partial last line waits for next poll."""
        stat = path.stat()
        if stat.st_ino != self.inode or stat.st_size < self.offset:
            self.inode, self.offset = stat.st_ino, 0
            self.requests.clear()
            self.credit_error_at = None
            self.incomplete = False
        with path.open("rb") as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                self.offset = stream.tell()
                try:
                    record = json.loads(line)
                    self._record(record)
                except (ValueError, TypeError, KeyError, AttributeError):
                    self.incomplete = True

    def _record(self, record: dict) -> None:
        if record.get("type") != "assistant":
            return
        msg = record.get("message") or {}
        if not isinstance(msg, dict):
            return
        at = timestamp(record["timestamp"])
        # Inspect API-error messages, not user prompts or tool output quoting an error.
        if record.get("isApiErrorMessage") or msg.get("model") == "<synthetic>":
            texts = [part.get("text", "") for part in msg.get("content", []) if isinstance(part, dict)]
            if any(error in text.lower() for text in texts for error in CREDIT_ERRORS):
                self.credit_error_at = max(self.credit_error_at or at, at)
            return
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            return
        key = record.get("requestId") or msg.get("id")
        if not key:
            self.incomplete = True
            return
        old = self.requests.get(key)
        if old is None or usage.get("output_tokens", 0) >= old[2].get("output_tokens", 0):
            self.requests[key] = (at, msg.get("model", ""), usage)


class CreditMonitor:
    """In-memory incremental observer. Never reads or writes credentials."""

    def __init__(self, root: Path):
        self.root = root
        self.logs: dict[Path, _Log] = {}

    def scan(self, snap: AccountsSnapshot, *, now: float) -> dict[str, CreditEstimate]:
        config = load_credit_accounts(self.root)
        accounts = {a.number: a for a in snap.accounts}
        estimates = {}
        live_paths = set()
        for number, entry in config.items():
            acc = accounts.get(number)
            parent = accounts.get(entry["group_with"])
            # Stale metadata must not attribute a replacement slot's spend to an email.
            if acc is None or parent is None or identity(acc) != entry["source_identity"]:
                continue
            if identity(parent) != entry.get("group_identity"):
                continue
            period_start, resets_at = entry["since"], None
            if entry.get("monthly_cycle"):
                period_start, resets_at = monthly_window(period_start, now)
            since = timestamp(period_start)
            requests = {}
            error_at = None
            incomplete = False
            directory = session_dir_for(self.root, number, acc.email) / "projects"
            paths = []
            walk_errors = []
            for folder, _directories, names in os.walk(directory, onerror=walk_errors.append):
                paths.extend(Path(folder) / name for name in names if name.endswith(".jsonl"))
            incomplete |= any(not isinstance(error, FileNotFoundError) for error in walk_errors)
            for path in paths:
                live_paths.add(path)
                log = self.logs.setdefault(path, _Log())
                try:
                    log.update(path)
                except OSError:
                    incomplete = True
                    continue
                incomplete |= log.incomplete
                if log.credit_error_at is not None and since <= log.credit_error_at <= now:
                    error_at = max(error_at or log.credit_error_at, log.credit_error_at)
                for key, request in log.requests.items():
                    if not since <= request[0] <= now:
                        continue
                    old = requests.get(key)
                    if old is None or request[2].get("output_tokens", 0) >= old[2].get("output_tokens", 0):
                        requests[key] = request
            spent = 0.0
            last_used = None
            for at, model, usage in requests.values():
                try:
                    cost = request_cost(model, usage)
                except (TypeError, AttributeError):
                    cost = None
                if cost is None:
                    incomplete = True
                else:
                    spent += cost
                last_used = max(last_used or at, at)
            estimates[number] = CreditEstimate(
                number, parent.number, parent.email, entry["monthly_credit"], period_start,
                spent, len(requests), last_used, error_at, incomplete,
                source_identity=tuple(identity(acc)), group_org_uuid=parent.org_uuid, resets_at=resets_at,
            )
        self.logs = {p: log for p, log in self.logs.items() if p in live_paths}
        return estimates
