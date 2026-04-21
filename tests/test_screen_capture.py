from pathlib import Path
from types import SimpleNamespace

import pytest

from haindy.linux.screen_capture import ScreenCapture


class _CompletedProcess:
    def __init__(self, stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_screen_capture_unique_filenames(monkeypatch, tmp_path: Path) -> None:
    resolution_manager = SimpleNamespace(viewport_size=lambda: (800, 600))
    monkeypatch.setattr(
        "haindy.linux.screen_capture.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    capture = ScreenCapture(
        resolution_manager=resolution_manager,
        screenshot_dir=tmp_path / "shots",
        display=":0",
        max_screenshots=5,
    )

    monkeypatch.setattr(
        "haindy.linux.screen_capture.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(stdout=b"png"),
    )

    _, path_a = capture.capture("desktop")
    _, path_b = capture.capture("desktop")

    assert path_a != path_b
    assert Path(path_a).exists()
    assert Path(path_b).exists()


def test_screen_capture_applies_retention(monkeypatch, tmp_path: Path) -> None:
    resolution_manager = SimpleNamespace(viewport_size=lambda: (800, 600))
    monkeypatch.setattr(
        "haindy.linux.screen_capture.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    capture = ScreenCapture(
        resolution_manager=resolution_manager,
        screenshot_dir=tmp_path / "shots",
        display=":0",
        max_screenshots=1,
    )

    monkeypatch.setattr(
        "haindy.linux.screen_capture.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(stdout=b"png"),
    )

    capture.capture("desktop")
    capture.capture("desktop")
    pngs = sorted((tmp_path / "shots").glob("*.png"))
    assert len(pngs) == 1


def test_screen_capture_falls_back_to_import(monkeypatch, tmp_path: Path) -> None:
    resolution_manager = SimpleNamespace(viewport_size=lambda: (800, 600))
    monkeypatch.setattr(
        "haindy.linux.screen_capture.shutil.which",
        lambda name: (
            "/usr/bin/ffmpeg"
            if name == "ffmpeg"
            else ("/usr/bin/import" if name == "import" else None)
        ),
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        if cmd[0] == "/usr/bin/ffmpeg":
            return _CompletedProcess(
                stdout=b"", returncode=1, stderr=b"x11grab not available"
            )
        return _CompletedProcess(stdout=b"png-data")

    monkeypatch.setattr("haindy.linux.screen_capture.subprocess.run", _fake_run)

    capture = ScreenCapture(
        resolution_manager=resolution_manager,
        screenshot_dir=tmp_path / "shots",
        display=":0",
    )

    screenshot, path = capture.capture("desktop")

    assert screenshot == b"png-data"
    assert Path(path).exists()
    assert calls[0][0] == "/usr/bin/ffmpeg"
    assert calls[1][0] == "/usr/bin/import"


def test_screen_capture_raises_when_no_backend(monkeypatch, tmp_path: Path) -> None:
    resolution_manager = SimpleNamespace(viewport_size=lambda: (800, 600))
    monkeypatch.setattr("haindy.linux.screen_capture.shutil.which", lambda name: None)
    capture = ScreenCapture(
        resolution_manager=resolution_manager,
        screenshot_dir=tmp_path / "shots",
        display=":0",
    )

    with pytest.raises(RuntimeError, match="No screen capture backend available"):
        capture.capture("desktop")
