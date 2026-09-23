"""Read the current Codex login's limits through Codex's local app-server.

No credentials are read or stored by claude-swap. The app-server uses the
existing Codex login and returns only usage windows and reset timestamps.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class CodexWindow:
    label: str
    pct: float
    resets_at: str | None


@dataclass(frozen=True)
class CodexUsage:
    windows: tuple[CodexWindow, ...] = ()
    status: str = ""
    plan: str = ""
    fetched_at: float | None = None


def _window_label(minutes: float) -> str:
    if minutes >= 1440 and minutes % 1440 == 0:
        return f"{minutes / 1440:g}d"
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes / 60:g}h"
    return f"{minutes:g}m"


def _parse_window(value: object, name: str = "") -> CodexWindow | None:
    if not isinstance(value, dict):
        return None
    pct, minutes, reset = (value.get(k) for k in ("usedPercent", "windowDurationMins", "resetsAt"))
    if not isinstance(pct, (float, int)) or isinstance(pct, bool):
        return None
    if not isinstance(minutes, (float, int)) or isinstance(minutes, bool) or minutes <= 0:
        return None
    resets_at = None
    if isinstance(reset, (int, float)) and not isinstance(reset, bool) and reset > 0:
        resets_at = datetime.fromtimestamp(reset, tz=timezone.utc).isoformat()
    label = _window_label(minutes)
    return CodexWindow(f"{name} {label}" if name else label, float(pct), resets_at)


def parse_rate_limits(result: dict) -> CodexUsage:
    """Keep only provider-defined windows; never invent a missing 5h/7d bar."""
    aggregate = result.get("rateLimits")
    if not isinstance(aggregate, dict):
        return CodexUsage(status="usage unavailable · run codex login if signed out")
    windows = []
    for key in ("primary", "secondary"):
        window = _parse_window(aggregate.get(key))
        if window:
            windows.append(window)
    by_id = result.get("rateLimitsByLimitId")
    if isinstance(by_id, dict):
        for limit_id, bucket in by_id.items():
            if (not isinstance(bucket, dict) or bucket == aggregate
                    or limit_id in {"codex", aggregate.get("limitId")}):
                continue
            name = bucket.get("limitName") or str(limit_id)
            for key in ("primary", "secondary"):
                window = _parse_window(bucket.get(key), name)
                if window:
                    windows.append(window)
    if not windows:
        return CodexUsage(status="usage unavailable · run codex login if signed out")
    return CodexUsage(tuple(windows), plan=str(result.get("planType") or ""), fetched_at=time.time())


def _read_lines(stdout, messages: queue.Queue) -> None:
    for line in stdout:
        try:
            messages.put(json.loads(line))
        except json.JSONDecodeError:
            continue
    messages.put(None)


def _response(messages: queue.Queue, request_id: int, deadline: float) -> dict | None:
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            message = messages.get(timeout=remaining)
        except queue.Empty:
            break
        if message is None:
            break
        if isinstance(message, dict) and message.get("id") == request_id:
            return message
    return None


def fetch_codex_usage(timeout: float = 15.0) -> CodexUsage:
    """One bounded, read-only app-server request, made off the UI thread."""
    codex = shutil.which("codex")
    if not codex:
        return CodexUsage(status="Codex CLI not found")
    try:
        process = subprocess.Popen(
            [codex, "app-server", "--stdio"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
    except OSError:
        return CodexUsage(status="Codex CLI could not start")
    deadline = time.monotonic() + timeout
    try:
        assert process.stdin is not None and process.stdout is not None
        messages: queue.Queue = queue.Queue()
        threading.Thread(target=_read_lines, args=(process.stdout, messages), daemon=True).start()
        def send(message: dict) -> None:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        send({"method": "initialize", "id": 1, "params": {
            "clientInfo": {"name": "claude_swap", "title": "Claude Swap", "version": "1"},
        }})
        initialized = _response(messages, 1, deadline)
        if not initialized or "result" not in initialized:
            return CodexUsage(status="Codex usage unavailable · run codex login if signed out")
        send({"method": "initialized"})
        send({"method": "account/rateLimits/read", "id": 2})
        response = _response(messages, 2, deadline)
        if not response or not isinstance(response.get("result"), dict):
            return CodexUsage(status="Codex usage unavailable · run codex login if signed out")
        return parse_rate_limits(response["result"])
    except (OSError, ValueError, OverflowError):
        return CodexUsage(status="Codex usage unavailable · run codex login if signed out")
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
