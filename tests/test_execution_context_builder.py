"""Tests for shared execution-context builders."""

from haindy.agents.situational_agent import SetupInstructions, SituationalAssessment
from haindy.runtime.execution_context_builder import build_execution_context_bundle


def test_build_execution_context_bundle_preserves_expected_shapes() -> None:
    assessment = SituationalAssessment(
        target_type="web",
        sufficient=True,
        setup=SetupInstructions(
            web_url="https://example.com",
            app_name="Example",
            launch_command="python app.py",
            maximize=False,
            adb_serial="emulator-5554",
            app_package="com.example.app",
            app_activity="MainActivity",
            adb_commands=["adb shell input keyevent 3"],
        ),
        notes=["Use staging credentials"],
    )

    bundle = build_execution_context_bundle(
        context_text="target: web",
        assessment=assessment,
        automation_backend="desktop",
    )

    assert bundle.planning_context == {
        "execution_context": "target: web",
        "target_type": "web",
        "automation_backend": "desktop",
        "web_url": "https://example.com",
        "app_name": "Example",
        "launch_command": "python app.py",
        "maximize": "False",
        "adb_serial": "emulator-5554",
        "app_package": "com.example.app",
        "app_activity": "MainActivity",
        "adb_commands": ["adb shell input keyevent 3"],
    }
    assert bundle.planning_cache_key_context == {"execution_context": "target: web"}
    assert bundle.test_context == {
        "execution_context": "target: web",
        "target_type": "web",
        "automation_backend": "desktop",
        "entry_setup": {
            "web_url": "https://example.com",
            "app_name": "Example",
            "launch_command": "python app.py",
            "maximize": False,
            "adb_serial": "emulator-5554",
            "app_package": "com.example.app",
            "app_activity": "MainActivity",
            "adb_commands": ["adb shell input keyevent 3"],
        },
        "setup_notes": ["Use staging credentials"],
    }
    assert bundle.initial_url == "https://example.com"
