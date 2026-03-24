"""Asynchronous ADB subprocess client with strict command validation."""

from __future__ import annotations

import asyncio
from asyncio import subprocess as aio_subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ADBCommandResult:
    """Result of a completed ADB command."""

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


class ADBCommandError(RuntimeError):
    """Raised when an ADB command returns a non-zero exit code."""

    def __init__(self, result: ADBCommandResult):
        stderr = result.stderr_text.strip()
        command = " ".join(result.command)
        message = f"ADB command failed (exit {result.returncode}): {command}" + (
            f" | stderr: {stderr}" if stderr else ""
        )
        super().__init__(message)
        self.result = result


class ADBClient:
    """Thin async wrapper around `adb` with timeout and command safety checks."""

    def __init__(
        self,
        adb_path: str = "adb",
        serial: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.adb_path = adb_path
        self.serial = self._normalize_serial(serial)
        self.timeout_seconds = max(float(timeout_seconds), 0.1)
        self._allowed_adb_binaries = {
            "adb",
            "adb.exe",
            Path(adb_path).name.lower(),
        }

    async def run_command(
        self,
        command: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> ADBCommandResult:
        """Run a fully-formed adb command with strict binary validation."""
        normalized = tuple(str(part) for part in command)
        self._validate_adb_command(normalized)

        try:
            process = await asyncio.create_subprocess_exec(
                *normalized,
                stdout=aio_subprocess.PIPE,
                stderr=aio_subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"ADB executable not found: {normalized[0]!r}") from exc

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
                f"ADB command timed out after {effective_timeout:.1f}s: {command_str}"
            ) from exc

        return ADBCommandResult(
            command=normalized,
            returncode=process.returncode or 0,
            stdout=stdout or b"",
            stderr=stderr or b"",
        )

    async def run_adb(
        self,
        *args: str,
        timeout_seconds: float | None = None,
        check: bool = True,
        serial: str | None = None,
    ) -> ADBCommandResult:
        """Run an adb command with optional serial override."""
        command: list[str] = [self.adb_path]
        effective_serial = self._normalize_serial(serial) or self.serial
        if effective_serial:
            command.extend(["-s", effective_serial])
        command.extend(str(part) for part in args)
        result = await self.run_command(command, timeout_seconds=timeout_seconds)
        if check and result.returncode != 0:
            raise ADBCommandError(result)
        return result

    async def list_devices(self) -> list[str]:
        """Return connected device serials in `device` state."""
        result = await self.run_adb("devices")
        devices: list[str] = []
        for raw_line in result.stdout_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("List of devices attached"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    async def resolve_serial(self, preferred_serial: str | None = None) -> str:
        """Resolve the target device serial, enforcing a single active device."""
        preferred = self._normalize_serial(preferred_serial)
        devices = await self.list_devices()
        if preferred:
            if preferred not in devices:
                available = ", ".join(devices) if devices else "<none>"
                raise RuntimeError(
                    f"Preferred adb device not found: {preferred}. Available: {available}"
                )
            return preferred

        if not devices:
            raise RuntimeError("No connected Android devices found via adb devices.")
        if len(devices) > 1:
            available = ", ".join(devices)
            raise RuntimeError(
                "Multiple Android devices connected. Set a preferred serial. "
                f"Available: {available}"
            )
        return devices[0]

    def _validate_adb_command(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("ADB command is empty.")
        executable_name = Path(str(command[0])).name.lower()
        if executable_name not in self._allowed_adb_binaries:
            raise ValueError(
                f"Only adb commands are allowed. Received executable: {command[0]!r}"
            )

    @staticmethod
    def _normalize_serial(serial: str | None) -> str | None:
        if serial is None:
            return None
        trimmed = serial.strip()
        return trimmed or None
