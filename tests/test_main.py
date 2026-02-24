"""Tests for desktop-first CLI interface."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.main import async_main, create_parser, read_context_file, read_plan_file, run_test


class TestCLIParser:
    def test_parser_creation(self) -> None:
        parser = create_parser()
        actions = {action.dest: action for action in parser._actions}
        assert "plan" in actions
        assert "context" in actions
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
async def test_run_test_blocks_on_insufficient_context() -> None:
    with patch("src.main.initialize_debug_logger") as mock_debug_logger, patch(
        "src.main._create_planning_agents"
    ) as mock_create_agents:
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
                "as_blocking_questions": lambda self: ["Missing required context: web_url"],
            },
        )()
        mock_create_agents.return_value = (triage, planner, situational)

        result = await run_test(
            requirements="Test requirement",
            context_text="no target",
            timeout=1,
        )

    assert result == 1
