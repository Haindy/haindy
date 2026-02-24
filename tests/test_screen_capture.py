from pathlib import Path
from types import SimpleNamespace

from src.desktop.screen_capture import ScreenCapture


class _CompletedProcess:
    def __init__(self, stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_screen_capture_unique_filenames(monkeypatch, tmp_path: Path) -> None:
    resolution_manager = SimpleNamespace(viewport_size=lambda: (800, 600))
    capture = ScreenCapture(
        resolution_manager=resolution_manager,
        screenshot_dir=tmp_path / "shots",
        display=":0",
        max_screenshots=5,
    )

    monkeypatch.setattr(
        "src.desktop.screen_capture.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(stdout=b"png"),
    )

    _, path_a = capture.capture("desktop")
    _, path_b = capture.capture("desktop")

    assert path_a != path_b
    assert Path(path_a).exists()
    assert Path(path_b).exists()


def test_screen_capture_applies_retention(monkeypatch, tmp_path: Path) -> None:
    resolution_manager = SimpleNamespace(viewport_size=lambda: (800, 600))
    capture = ScreenCapture(
        resolution_manager=resolution_manager,
        screenshot_dir=tmp_path / "shots",
        display=":0",
        max_screenshots=1,
    )

    monkeypatch.setattr(
        "src.desktop.screen_capture.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(stdout=b"png"),
    )

    capture.capture("desktop")
    capture.capture("desktop")
    pngs = sorted((tmp_path / "shots").glob("*.png"))
    assert len(pngs) == 1
