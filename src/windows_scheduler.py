"""Explicit, one-time Windows .exe launches using Windows Task Scheduler.

No elevated privileges requested, no arbitrary shell command input, no running
applications without a user-initiated registration and GUI confirmation.
Tasks run only in the current user's interactive Windows session (PC awake and
user signed in); the original .exe path must still exist at execution time.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

TASK_PREFIX = "SunPilotAI_"
TASK_PATTERN = re.compile(r"^SunPilotAI_[a-f0-9]{12}$")
MARKER = "SUNPILOT_JSON:"


class ScheduleError(RuntimeError):
    pass


def validate_request(executable: str | Path, instant_utc: datetime, *,
                     now_utc: datetime | None = None) -> tuple[Path, datetime]:
    """Require a real .exe and a UTC-aware future start instant."""
    path = Path(executable).expanduser()
    if path.suffix.lower() != ".exe":
        raise ValueError("Choose a Windows .exe program (not a script or shortcut).")
    if not path.is_file():
        raise FileNotFoundError(f"The selected program does not exist:\n{path}")
    if instant_utc.tzinfo is None or instant_utc.utcoffset() is None:
        raise ValueError("The scheduled time must include a timezone.")
    moment = instant_utc.astimezone(timezone.utc)
    current = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if moment < current + timedelta(minutes=2):
        raise ValueError("Choose a start time at least 2 minutes in the future.")
    return path.resolve(), moment


def _encode_ps(text: str) -> str:
    return base64.b64encode(text.encode("utf-16-le")).decode("ascii")


def _ps_with_payload(script_body: str, payload: dict) -> str:
    """Payload is transported as base64 JSON; no paths become PowerShell code."""
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return (
        "$ErrorActionPreference='Stop'\n"
        "try {\n"
        f"  $p = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}')) | ConvertFrom-Json\n"
        f"  {script_body}\n"
        "} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }\n"
    )


def _invoke(script: str, *, timeout: int = 40):
    if sys.platform != "win32":
        raise ScheduleError("Windows Task Scheduler is available only on Windows.")
    cmd = [
        "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
        "-EncodedCommand", _encode_ps(script),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScheduleError(f"Cannot contact Windows Task Scheduler: {exc}") from exc
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "Windows rejected this scheduled task").strip()
        raise ScheduleError(msg[:1600])
    output = result.stdout or ""
    for line in reversed(output.splitlines()):
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER):])
            except json.JSONDecodeError as exc:
                raise ScheduleError("Task Scheduler returned invalid response data") from exc
    raise ScheduleError("Task Scheduler did not confirm the operation.")


def register_launch(executable: str | Path, when_utc: datetime, *,
                    now_utc: datetime | None = None) -> dict:
    """Create one launch at Windows local wall time, after explicit GUI consent.

    Schedule is recorded on the current computer. It will not run if the user
    has signed out or the PC is asleep at trigger time. No admin rights or
    password are requested.
    """
    path, moment = validate_request(executable, when_utc, now_utc=now_utc)
    local = moment.astimezone().replace(tzinfo=None)
    # Keep ISO local wall time, parsed invariantly inside PowerShell.
    task_name = TASK_PREFIX + uuid.uuid4().hex[:12]
    payload = {
        "exe": str(path), "directory": str(path.parent),
        "run_local": local.strftime("%Y-%m-%dT%H:%M:%S"), "name": task_name,
    }
    script = _ps_with_payload(r"""
    $at = [datetime]::ParseExact($p.run_local, 'yyyy-MM-ddTHH:mm:ss', [Globalization.CultureInfo]::InvariantCulture)
    if ($at -le (Get-Date).AddMinutes(1)) { throw 'The scheduled time is too close or already passed.' }
    $action = New-ScheduledTaskAction -Execute $p.exe -WorkingDirectory $p.directory
    $trigger = New-ScheduledTaskTrigger -Once -At $at
    $userId = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
    Register-ScheduledTask -TaskName $p.name -TaskPath '\' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'SunPilot AI: one-time user-approved executable launch' -ErrorAction Stop | Out-Null
    [Console]::Out.WriteLine('SUNPILOT_JSON:' + (ConvertTo-Json -InputObject @{ name=$p.name; run_local=$p.run_local; exe=$p.exe } -Compress))
    """, payload)
    result = _invoke(script)
    if not isinstance(result, dict) or result.get("name") != task_name:
        raise ScheduleError("The registered task was not confirmed.")
    return result


def list_launches() -> list[dict]:
    script = _ps_with_payload(r"""
    $tasks = @(Get-ScheduledTask -TaskPath '\' -ErrorAction Stop | Where-Object { $_.TaskName -match '^SunPilotAI_[0-9a-f]{12}$' })
    $rows = @()
    foreach ($task in $tasks) {
        $action = @($task.Actions)[0]
        $trigger = @($task.Triggers)[0]
        $rows += [PSCustomObject]@{
            name = [string]$task.TaskName
            exe = [string]$action.Execute
            run_local = [string]$trigger.StartBoundary
            status = [string]$task.State
        }
    }
    [Console]::Out.WriteLine('SUNPILOT_JSON:' + (ConvertTo-Json -InputObject @($rows) -Compress -Depth 4))
    """, {})
    result = _invoke(script)
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    if not isinstance(result, list):
        raise ScheduleError("Unexpected task list response")
    return [t for t in result if isinstance(t, dict) and TASK_PATTERN.fullmatch(t.get("name", ""))]


def cancel_launch(task_name: str) -> None:
    if not TASK_PATTERN.fullmatch(task_name):
        raise ValueError("Invalid SunPilot scheduled task identifier")
    script = _ps_with_payload(r"""
    Unregister-ScheduledTask -TaskName $p.name -TaskPath '\' -Confirm:$false -ErrorAction Stop
    [Console]::Out.WriteLine('SUNPILOT_JSON:' + (ConvertTo-Json -InputObject @{ removed=$p.name } -Compress))
    """, {"name": task_name})
    result = _invoke(script)
    if not isinstance(result, dict) or result.get("removed") != task_name:
        raise ScheduleError("Windows did not confirm cancellation")
