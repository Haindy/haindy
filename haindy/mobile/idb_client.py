"""Asynchronous idb subprocess client with strict command validation."""

from __future__ import annotations

import asyncio
import json
import re
from asyncio import subprocess as aio_subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

_UDID_PATTERN = re.compile(
    # Standard UUID: 8-4-4-4-12
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    # Device UDID with dashes (e.g. 00008120-001635162EEB401E)
    r"|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{12,16}"
    # Plain hex string (no dashes)
    r"|[0-9A-Fa-f]{16,64}"
)


@dataclass(frozen=True)
class IDBCommandResult:
    """Result of a completed idb command."""

    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="ignore")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="ignore")


class IDBCommandError(RuntimeError):
    """Raised when an idb command returns a non-zero exit code."""

    def __init__(self, result: IDBCommandResult):
        stderr = result.stderr_text.strip()
        command = " ".join(result.command)
        message = f"idb command failed (exit {result.returncode}): {command}" + (
            f" | stderr: {stderr}" if stderr else ""
        )
        super().__init__(message)
        self.result = result


@runtime_checkable
class IDBClientProtocol(Protocol):
    """Structural protocol for idb client objects (real or stub)."""

    udid: str | None
    timeout_seconds: float

    async def run_idb(
        self,
        *args: str,
        timeout_seconds: float | None = None,
        check: bool = True,
        udid: str | None = None,
    ) -> IDBCommandResult: ...

    async def resolve_udid(self, preferred_udid: str | None = None) -> str: ...


class IDBClient:
    """Thin async wrapper around `idb` with timeout and command safety checks."""

    def __init__(
        self,
        idb_path: str = "idb",
        udid: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.idb_path = idb_path
        self.udid = self._normalize_udid(udid)
        self.timeout_seconds = max(float(timeout_seconds), 0.1)
        self._allowed_idb_binaries = {
            "idb",
            "idb.exe",
            Path(idb_path).name.lower(),
        }

    async def run_command(
        self,
        command: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> IDBCommandResult:
        """Run a fully-formed idb command with strict binary validation."""
        normalized = tuple(str(part) for part in command)
        self._validate_idb_command(normalized)

        try:
            process = await asyncio.create_subprocess_exec(
                *normalized,
                stdout=aio_subprocess.PIPE,
                stderr=aio_subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"idb executable not found: {normalized[0]!r}") from exc

        effective_timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else max(float(timeout_seconds), 0.1)
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            command_str = " ".join(normalized)
            raise TimeoutError(
                f"idb command timed out after {effective_timeout:.1f}s: {command_str}"
            ) from exc

        return IDBCommandResult(
            command=normalized,
            returncode=process.returncode or 0,
            stdout=stdout or b"",
            stderr=stderr or b"",
        )

    async def run_idb(
        self,
        *args: str,
        timeout_seconds: float | None = None,
        check: bool = True,
        udid: str | None = None,
    ) -> IDBCommandResult:
        """Run an idb command, appending --udid if set."""
        command: list[str] = [self.idb_path]
        command.extend(str(part) for part in args)
        effective_udid = self._normalize_udid(udid) or self.udid
        if effective_udid:
            command.extend(["--udid", effective_udid])
        result = await self.run_command(command, timeout_seconds=timeout_seconds)
        if check and result.returncode != 0:
            raise IDBCommandError(result)
        return result

    async def list_targets(self) -> list[dict[str, object]]:
        """Return info dicts for all known targets (connected or booted)."""
        result = await self.run_idb("list-targets", "--json", udid=None, check=False)
        text = result.stdout_text.strip()
        if text:
            # Try NDJSON (one JSON object per line — idb 1.x format)
            targets: list[dict[str, object]] = []
            all_parsed = True
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        targets.append(obj)
                    else:
                        all_parsed = False
                        break
                except (json.JSONDecodeError, ValueError):
                    all_parsed = False
                    break
            if all_parsed and targets:
                return targets
            # Try JSON array (future idb versions)
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [t for t in parsed if isinstance(t, dict)]
            except (json.JSONDecodeError, ValueError):
                pass
        return self._parse_targets_text(result.stdout_text)

    async def resolve_udid(self, preferred_udid: str | None = None) -> str:
        """Resolve the target device UDID, enforcing a single active device."""
        preferred = self._normalize_udid(preferred_udid)
        targets = await self.list_targets()
        available = [
            str(t.get("udid", ""))
            for t in targets
            if str(t.get("state", "")).lower() in {"booted", "connected"}
            and t.get("udid")
        ]

        if preferred:
            if preferred not in available:
                available_str = ", ".join(available) if available else "<none>"
                raise RuntimeError(
                    f"Preferred idb device not found: {preferred}. "
                    f"Available: {available_str}"
                )
            return preferred

        if not available:
            raise RuntimeError(
                "No connected or booted iOS devices/simulators found via idb list-targets."
            )
        if len(available) > 1:
            available_str = ", ".join(available)
            raise RuntimeError(
                "Multiple iOS devices/simulators found. Set a preferred UDID. "
                f"Available: {available_str}"
            )
        return available[0]

    def _validate_idb_command(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("idb command is empty.")
        executable_name = Path(str(command[0])).name.lower()
        if executable_name not in self._allowed_idb_binaries:
            raise ValueError(
                f"Only idb commands are allowed. Received executable: {command[0]!r}"
            )

    @staticmethod
    def _normalize_udid(udid: str | None) -> str | None:
        if udid is None:
            return None
        trimmed = udid.strip()
        return trimmed or None

    @staticmethod
    def _parse_targets_text(text: str) -> list[dict[str, object]]:
        """Fallback parser for non-JSON idb list-targets output.

        Expects lines like: Name | UDID | State | Type | OS
        """
        targets: list[dict[str, object]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                udid_candidate = parts[1] if len(parts) > 1 else ""
                state_candidate = parts[2] if len(parts) > 2 else ""
                if _UDID_PATTERN.match(udid_candidate):
                    targets.append(
                        {
                            "name": parts[0],
                            "udid": udid_candidate,
                            "state": state_candidate,
                            "type": parts[3] if len(parts) > 3 else "",
                        }
                    )
        return targets
