"""Tests for desktop-first CLI interface."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.main import (
    _create_coordinator_stack,
    async_main,
    create_parser,
    read_context_file,
    read_plan_file,
    run_test,
)


class TestCLIParser:
    def test_parser_creation(self) -> None:
        parser = create_parser()
        actions = {action.dest: action for action in parser._actions}
        assert "plan" in actions
        assert "context" in actions
        assert "mobile" in actions
        assert "test_api" in actions
        assert "version" in actions

    def test_mutually_exclusive_inputs(self) -> None:
        parser = create_parser()
        parser.parse_args(["--plan", "test.md", "--context", "ctx.txt"])
        parser.parse_args(["--test-api"])
        parser.parse_args(["--version"])

        with pytest.raises(SystemExit):
            parser.parse_args(["--plan", "test.md", "--test-api"])


class TestFileLoading:
    @pytest.mark.asyncio
    async def test_read_plan_file_not_found(self) -> None:
        with pytest.raises(SystemExit):
            await read_plan_file(Path("missing.md"))

    @pytest.mark.asyncio
    async def test_read_context_file_not_found(self) -> None:
        with pytest.raises(SystemExit):
            await read_context_file(Path("missing.txt"))

    @pytest.mark.asyncio
    async def test_read_context_file_empty(self, tmp_path: Path) -> None:
        context_file = tmp_path / "context.txt"
        context_file.write_text("\n")
        with pytest.raises(SystemExit):
            await read_context_file(context_file)


class TestMainFlow:
    @pytest.mark.asyncio
    async def test_requires_context_when_plan_supplied(self, capsys):
        result = await async_main(["--plan", "requirements.md"])
        captured = capsys.readouterr()
        assert result == 1
        assert "--context is required" in captured.out

    @pytest.mark.asyncio
    async def test_invokes_run_test_with_plan_and_context(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.txt"
        context_file = tmp_path / "context.txt"
        plan_file.write_text("Test requirement")
        context_file.write_text("target: web\nurl: https://example.com")

        with patch("src.main.run_test", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0
            result = await async_main(
                ["--plan", str(plan_file), "--context", str(context_file)]
            )

        assert result == 0
        mock_run.assert_awaited_once()
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requirements"] == "Test requirement"
        assert "https://example.com" in kwargs["context_text"]

    @pytest.mark.asyncio
    async def test_mobile_flag_forces_mobile_backend(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.txt"
        context_file = tmp_path / "context.txt"
        plan_file.write_text("Test requirement")
        context_file.write_text("target_type: mobile_adb")

        with patch("src.main.run_test", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 0
            result = await async_main(
                ["--plan", str(plan_file), "--context", str(context_file), "--mobile"]
            )

        assert result == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["automation_backend"] == "mobile_adb"


@pytest.mark.asyncio
async def test_run_test_blocks_on_insufficient_context() -> None:
    with (
        patch("src.main.initialize_debug_logger") as mock_debug_logger,
        patch("src.main._create_planning_agents") as mock_create_agents,
    ):
        debug_instance = type("Debug", (), {"reports_dir": Path("reports")})()
        mock_debug_logger.return_value = debug_instance

        triage = AsyncMock()
        planner = AsyncMock()
        situational = AsyncMock()
        situational.assess_context.return_value = type(
            "Assessment",
            (),
            {
                "sufficient": False,
                "notes": ["Need a URL"],
                "as_blocking_questions": lambda self: [
                    "Missing required context: web_url"
                ],
            },
        )()
        mock_create_agents.return_value = (triage, planner, situational)

        result = await run_test(
            requirements="Test requirement",
            context_text="no target",
            timeout=1,
        )

    assert result == 1


@pytest.mark.asyncio
async def test_run_test_mobile_backend_rejects_non_mobile_assessment() -> None:
    with (
        patch("src.main.initialize_debug_logger") as mock_debug_logger,
        patch("src.main._create_planning_agents") as mock_create_agents,
    ):
        debug_instance = type("Debug", (), {"reports_dir": Path("reports")})()
        mock_debug_logger.return_value = debug_instance

        triage = AsyncMock()
        planner = AsyncMock()
        situational = AsyncMock()
        situational.assess_context.return_value = type(
            "Assessment",
            (),
            {
                "sufficient": True,
                "target_type": "desktop_app",
                "notes": [],
                "setup": type(
                    "Setup",
                    (),
                    {
                        "web_url": "",
                        "app_name": "",
                        "launch_command": "",
                        "maximize": True,
                        "adb_serial": "",
                        "app_package": "",
                        "app_activity": "",
                        "adb_commands": [],
                    },
                )(),
            },
        )()
        mock_create_agents.return_value = (triage, planner, situational)

        result = await run_test(
            requirements="Test requirement",
            context_text="desktop context",
            timeout=1,
            automation_backend="mobile_adb",
        )

    assert result == 1


@pytest.mark.asyncio
async def test_create_coordinator_stack_stops_desktop_on_start_failure() -> None:
    created: list[object] = []

    class FailingDesktopController:
        def __init__(self) -> None:
            self.start = AsyncMock(side_effect=RuntimeError("desktop start failed"))
            self.stop = AsyncMock()
            self.driver = object()
            created.append(self)

    with patch("src.main.DesktopController", FailingDesktopController):
        with pytest.raises(RuntimeError, match="desktop start failed"):
            await _create_coordinator_stack(max_steps=1)

    controller = created[0]
    assert isinstance(controller, FailingDesktopController)
    controller.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_coordinator_stack_uses_mobile_controller() -> None:
    created: list[object] = []

    class MobileControllerStub:
        def __init__(self) -> None:
            self.start = AsyncMock()
            self.stop = AsyncMock()
            self.driver = object()
            created.append(self)

    coordinator = type("CoordinatorStub", (), {"initialize": AsyncMock()})()

    with (
        patch("src.main.MobileController", MobileControllerStub),
        patch("src.main.WorkflowCoordinator", return_value=coordinator),
    ):
        controller, _ = await _create_coordinator_stack(
            max_steps=1, backend="mobile_adb"
        )

    assert isinstance(controller, MobileControllerStub)
    assert isinstance(created[0], MobileControllerStub)
    created[0].start.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_test_stops_desktop_even_if_coordinator_cleanup_fails(
    tmp_path: Path,
) -> None:
    debug_instance = type("Debug", (), {"reports_dir": tmp_path})()
    triage_agent = AsyncMock()
    planner = AsyncMock()
    situational = AsyncMock()
    situational.assess_context.return_value = type(
        "Assessment",
        (),
        {
            "sufficient": True,
            "target_type": "desktop_app",
            "setup": type(
                "Setup",
                (),
                {
                    "web_url": "",
                    "app_name": "KeenBench",
                    "launch_command": "",
                    "maximize": True,
                    "adb_serial": "",
                    "app_package": "",
                    "app_activity": "",
                    "adb_commands": [],
                },
            )(),
            "notes": [],
        },
    )()
    situational.prepare_entrypoint = AsyncMock()

    desktop_controller = type("DesktopControllerStub", (), {})()
    desktop_controller.driver = object()
    desktop_controller.stop = AsyncMock()

    coordinator = type("CoordinatorStub", (), {})()
    coordinator.cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    coordinator.get_action_agent = lambda: None
    mock_scope_pipeline = AsyncMock(return_value=(object(), object()))

    with (
        patch("src.main.initialize_debug_logger", return_value=debug_instance),
        patch(
            "src.main._create_planning_agents",
            return_value=(triage_agent, planner, situational),
        ),
        patch(
            "src.main.run_scope_triage_and_plan",
            new=mock_scope_pipeline,
        ),
        patch(
            "src.main._create_coordinator_stack",
            new=AsyncMock(return_value=(desktop_controller, coordinator)),
        ),
        patch(
            "src.main._run_with_timeout",
            new=AsyncMock(side_effect=RuntimeError("runner failed")),
        ),
    ):
        result = await run_test(
            requirements="Test requirement",
            context_text="desktop context",
            timeout=1,
        )

    assert result == 1
    assert mock_scope_pipeline.await_count == 1
    kwargs = mock_scope_pipeline.await_args.kwargs
    assert kwargs["cache_key_context"] == {"execution_context": "desktop context"}
    coordinator.cleanup.assert_awaited_once()
    desktop_controller.stop.assert_awaited_once()
