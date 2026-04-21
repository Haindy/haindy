"""Tests for desktop-first CLI interface."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haindy.main import (
    _create_coordinator_stack,
    _validate_auth_for_run,
    async_main,
    create_parser,
    read_context_file,
    read_plan_file,
    run_test,
)


class TestCLIParser:
    def test_parser_creation(self) -> None:
        parser = create_parser()
        subcommands = parser._subparsers._actions[-1].choices
        assert "run" in subcommands
        assert "auth" in subcommands
        assert "config" in subcommands
        assert "doctor" in subcommands
        assert "setup" in subcommands
        assert "version" in subcommands
        assert "test-api" in subcommands
        assert "provider" in subcommands
        assert "session" in subcommands
        assert "act" in subcommands
        assert "screenshot" in subcommands
        assert "test" in subcommands
        assert "test-status" in subcommands
        assert "explore" in subcommands
        assert "explore-status" in subcommands

    def test_provider_list_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["provider", "list"])
        assert args.command == "provider"
        assert args.provider_command == "list"

    def test_provider_set_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["provider", "set", "anthropic"])
        assert args.command == "provider"
        assert args.provider_command == "set"
        assert args.provider == "anthropic"

    def test_provider_set_computer_use_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["provider", "set-computer-use", "google"])
        assert args.command == "provider"
        assert args.provider_command == "set-computer-use"
        assert args.provider == "google"

    def test_provider_set_model_subcommand(self) -> None:
        parser = create_parser()
        args = parser.parse_args(
            ["provider", "set-model", "google", "gemini-3-flash-preview"]
        )
        assert args.command == "provider"
        assert args.provider_command == "set-model"
        assert args.provider == "google"
        assert args.model == "gemini-3-flash-preview"
        assert args.computer_use is False

    def test_provider_set_model_computer_use_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(
            [
                "provider",
                "set-model",
                "google",
                "gemini-3-flash-preview",
                "--computer-use",
            ]
        )
        assert args.provider_command == "set-model"
        assert args.computer_use is True

    def test_run_subcommand_args(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["run", "--plan", "test.md", "--context", "ctx.txt"])
        assert args.command == "run"
        assert args.plan == Path("test.md")
        assert args.context == Path("ctx.txt")

    def test_auth_subcommands(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["auth", "status"])
        assert args.command == "auth"
        assert args.auth_command == "status"

        args = parser.parse_args(["auth", "login", "openai"])
        assert args.command == "auth"
        assert args.auth_command == "login"
        assert args.provider == "openai"

        args = parser.parse_args(["auth", "clear", "openai-codex"])
        assert args.command == "auth"
        assert args.auth_command == "clear"
        assert args.provider == "openai-codex"

    def test_config_subcommands(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["config", "show"])
        assert args.command == "config"
        assert args.config_command == "show"

        args = parser.parse_args(["config", "migrate", "/path/to/.env"])
        assert args.command == "config"
        assert args.config_command == "migrate"
        assert args.dotenv_path == "/path/to/.env"

    def test_setup_non_interactive(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["setup", "--non-interactive"])
        assert args.command == "setup"
        assert args.non_interactive is True

    def test_run_mobile_flags(self) -> None:
        parser = create_parser()
        args = parser.parse_args(
            ["run", "--plan", "p.md", "--context", "c.txt", "--mobile"]
        )
        assert args.mobile is True

        args = parser.parse_args(
            ["run", "--plan", "p.md", "--context", "c.txt", "--ios"]
        )
        assert args.ios is True


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


class TestFirstRunGate:
    @pytest.mark.asyncio
    async def test_gate_blocks_run_when_marker_absent(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.txt"
        context_file = tmp_path / "context.txt"
        plan_file.write_text("Test requirement")
        context_file.write_text("target: web\nurl: https://example.com")

        with (
            patch("haindy.main._SETUP_MARKER", tmp_path / "does_not_exist"),
            pytest.raises(SystemExit) as exc_info,
        ):
            await async_main(
                ["run", "--plan", str(plan_file), "--context", str(context_file)]
            )

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_gate_bypassed_for_version(self, tmp_path: Path) -> None:
        with patch("haindy.main._SETUP_MARKER", tmp_path / "does_not_exist"):
            result = await async_main(["version"])
        assert result == 0

    @pytest.mark.asyncio
    async def test_gate_bypassed_for_doctor(self, tmp_path: Path) -> None:
        mock_module = MagicMock()
        mock_module.run_doctor = lambda: 0
        with (
            patch("haindy.main._SETUP_MARKER", tmp_path / "does_not_exist"),
            patch.dict("sys.modules", {"haindy.cli.doctor": mock_module}),
        ):
            result = await async_main(["doctor"])
        assert result == 0

    @pytest.mark.asyncio
    async def test_gate_bypassed_for_setup(self, tmp_path: Path) -> None:
        mock_module = MagicMock()
        mock_module.run_setup_wizard = AsyncMock(return_value=0)
        with (
            patch("haindy.main._SETUP_MARKER", tmp_path / "does_not_exist"),
            patch.dict("sys.modules", {"haindy.cli.setup_wizard": mock_module}),
        ):
            result = await async_main(["setup"])
        assert result == 0
        mock_module.run_setup_wizard.assert_awaited_once_with(non_interactive=False)


@pytest.mark.asyncio
async def test_setup_wizard_awaits_auth_login_instead_of_nesting_event_loop() -> None:
    from haindy.cli import setup_wizard

    prompts = iter([True, False, False])

    with (
        patch.object(setup_wizard, "_install_skills"),
        patch.object(setup_wizard, "run_doctor", side_effect=[1, 1]),
        patch.object(
            setup_wizard.Confirm,
            "ask",
            side_effect=lambda *args, **kwargs: next(prompts),
        ),
        patch("haindy.auth.credentials.get_api_key", return_value=""),
        patch(
            "haindy.cli.auth_commands.handle_auth_login", new=AsyncMock(return_value=0)
        ) as mock_login,
        patch("haindy.cli.setup_wizard.shutil.which", return_value=None),
        patch("haindy.cli.setup_wizard.write_settings_file"),
        patch("haindy.cli.setup_wizard.Path.touch"),
    ):
        result = await setup_wizard.run_setup_wizard()

    assert result == 1
    mock_login.assert_awaited_once_with("openai")


class TestMainFlow:
    @pytest.mark.asyncio
    async def test_requires_context_when_plan_supplied(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        marker = tmp_path / "setup_complete"
        marker.touch()
        with patch("haindy.main._SETUP_MARKER", marker):
            result = await async_main(["run", "--plan", "requirements.md"])
        captured = capsys.readouterr()
        assert result == 1
        assert "--context is required" in captured.out

    @pytest.mark.asyncio
    async def test_invokes_run_test_with_plan_and_context(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.txt"
        context_file = tmp_path / "context.txt"
        plan_file.write_text("Test requirement")
        context_file.write_text("target: web\nurl: https://example.com")
        marker = tmp_path / "setup_complete"
        marker.touch()

        with (
            patch("haindy.main._SETUP_MARKER", marker),
            patch("haindy.main._validate_auth_for_run", return_value=[]),
            patch("haindy.main.run_test", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = 0
            result = await async_main(
                ["run", "--plan", str(plan_file), "--context", str(context_file)]
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
        marker = tmp_path / "setup_complete"
        marker.touch()

        with (
            patch("haindy.main._SETUP_MARKER", marker),
            patch("haindy.main._validate_auth_for_run", return_value=[]),
            patch("haindy.main.run_test", new_callable=AsyncMock) as mock_run,
        ):
            mock_run.return_value = 0
            result = await async_main(
                [
                    "run",
                    "--plan",
                    str(plan_file),
                    "--context",
                    str(context_file),
                    "--mobile",
                ]
            )

        assert result == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["automation_backend"] == "mobile_adb"

    @pytest.mark.asyncio
    async def test_auth_status_dispatches_without_plan(self) -> None:
        marker = Path.home() / ".haindy" / "setup_complete"
        with (
            patch("haindy.main._SETUP_MARKER", marker),
            patch("haindy.main._is_setup_complete", return_value=True),
            patch(
                "haindy.main.handle_auth_status",
                new=AsyncMock(return_value=0),
            ) as mock_auth_status,
        ):
            result = await async_main(["auth", "status"])

        assert result == 0
        mock_auth_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_version_command_reports_installed_package_version(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("haindy.main.package_version", return_value="9.9.9"):
            result = await async_main(["version"])

        captured = capsys.readouterr()
        assert result == 0
        assert "Version: " in captured.out
        assert "9.9.9" in captured.out

    @pytest.mark.asyncio
    async def test_tool_call_cli_commands_dispatch_through_unified_parser(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch(
            "haindy.main.dispatch_tool_call_args",
            new=AsyncMock(
                return_value=(
                    SimpleNamespace(model_dump_json=lambda: '{"status":"success"}'),
                    0,
                )
            ),
        ) as mock_dispatch:
            result = await async_main(["session", "list"])

        captured = capsys.readouterr()
        assert result == 0
        assert captured.out.strip() == '{"status":"success"}'
        mock_dispatch.assert_awaited_once()
        parsed_args = mock_dispatch.await_args.args[0]
        assert parsed_args.command == "session"
        assert parsed_args.tool_command == "session"

    @pytest.mark.asyncio
    async def test_tool_call_daemon_commands_dispatch_before_legacy_parser(
        self,
    ) -> None:
        with patch(
            "haindy.main.run_tool_call_daemon_cli",
            new=AsyncMock(return_value=0),
        ) as mock_daemon:
            result = await async_main(
                [
                    "__tool_call_daemon",
                    "--session-id",
                    "abc123",
                    "--backend",
                    "desktop",
                ]
            )

        assert result == 0
        mock_daemon.assert_awaited_once_with(
            [
                "__tool_call_daemon",
                "--session-id",
                "abc123",
                "--backend",
                "desktop",
            ]
        )


@pytest.mark.asyncio
async def test_run_test_blocks_on_insufficient_context() -> None:
    with (
        patch("haindy.main.initialize_debug_logger") as mock_debug_logger,
        patch("haindy.main._create_planning_agents") as mock_create_agents,
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
        patch("haindy.main.initialize_debug_logger") as mock_debug_logger,
        patch("haindy.main._create_planning_agents") as mock_create_agents,
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
async def test_run_test_rejects_openai_cu_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "haindy.main.get_settings",
        lambda: SimpleNamespace(
            cu_provider="openai",
            openai_api_key="",
        ),
    )

    result = await run_test(
        requirements="Test requirement",
        context_text="desktop context",
        timeout=1,
    )

    assert result == 1


@pytest.mark.asyncio
async def test_create_coordinator_stack_stops_desktop_on_start_failure() -> None:
    created: list[object] = []

    class FailingController:
        def __init__(self) -> None:
            self.start = AsyncMock(side_effect=RuntimeError("desktop start failed"))
            self.stop = AsyncMock()
            self.driver = object()
            created.append(self)

    with (
        patch("haindy.main.sys") as mock_sys,
        patch("haindy.main.DesktopController", FailingController),
    ):
        mock_sys.platform = "linux"
        with pytest.raises(RuntimeError, match="desktop start failed"):
            await _create_coordinator_stack(max_steps=1)

    controller = created[0]
    assert isinstance(controller, FailingController)
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
        patch("haindy.main.MobileController", MobileControllerStub),
        patch("haindy.main.WorkflowCoordinator", return_value=coordinator),
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
        patch("haindy.main.initialize_debug_logger", return_value=debug_instance),
        patch(
            "haindy.main._create_planning_agents",
            return_value=(triage_agent, planner, situational),
        ),
        patch(
            "haindy.main.run_scope_triage_and_plan",
            new=mock_scope_pipeline,
        ),
        patch(
            "haindy.main._create_coordinator_stack",
            new=AsyncMock(return_value=(desktop_controller, coordinator)),
        ),
        patch(
            "haindy.main._run_with_timeout",
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
    assert mock_scope_pipeline.await_args is not None
    kwargs = mock_scope_pipeline.await_args.kwargs
    assert kwargs["cache_key_context"] == {"execution_context": "desktop context"}
    coordinator.cleanup.assert_awaited_once()
    desktop_controller.stop.assert_awaited_once()


class TestValidateAuthForRun:
    """Tests for _validate_auth_for_run."""

    def _make_settings(self, **kwargs):
        defaults = {
            "openai_api_key": "",
            "anthropic_api_key": "",
            "vertex_api_key": "",
            "agent_provider": "openai",
            "cu_provider": "openai",
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def _make_codex_status(self, *, oauth_connected=False, oauth_expired=False):
        return SimpleNamespace(
            oauth_connected=oauth_connected,
            oauth_expired=oauth_expired,
        )

    def test_anthropic_agent_provider_with_no_key_reports_issue(self):
        settings = self._make_settings(
            agent_provider="anthropic",
            anthropic_api_key="",
            cu_provider="openai",
            openai_api_key="sk-openai-key",
        )
        with patch("haindy.main.OpenAIAuthManager") as mock_mgr_cls:
            mock_mgr_cls.return_value.get_status.return_value = (
                self._make_codex_status()
            )
            issues = _validate_auth_for_run(settings)
        assert any("anthropic" in i.lower() for i in issues)

    def test_anthropic_agent_provider_with_key_no_issue(self):
        settings = self._make_settings(
            agent_provider="anthropic",
            anthropic_api_key="sk-ant-123",
            cu_provider="openai",
            openai_api_key="sk-openai-key",
        )
        with patch("haindy.main.OpenAIAuthManager") as mock_mgr_cls:
            mock_mgr_cls.return_value.get_status.return_value = (
                self._make_codex_status()
            )
            issues = _validate_auth_for_run(settings)
        # No agent provider issue (but may still have CU issue)
        agent_issues = [
            i
            for i in issues
            if "planning" in i.lower()
            or "anthropic" in i.lower()
            and "computer-use" not in i.lower()
        ]
        assert len(agent_issues) == 0

    def test_google_agent_provider_with_no_key_reports_issue(self):
        settings = self._make_settings(
            agent_provider="google",
            vertex_api_key="",
            cu_provider="openai",
            openai_api_key="sk-openai-key",
        )
        with patch("haindy.main.OpenAIAuthManager") as mock_mgr_cls:
            mock_mgr_cls.return_value.get_status.return_value = (
                self._make_codex_status()
            )
            issues = _validate_auth_for_run(settings)
        assert any("google" in i.lower() for i in issues)

    def test_google_agent_provider_with_key_no_agent_issue(self):
        settings = self._make_settings(
            agent_provider="google",
            vertex_api_key="key123",
            cu_provider="openai",
            openai_api_key="sk-openai-key",
        )
        with patch("haindy.main.OpenAIAuthManager") as mock_mgr_cls:
            mock_mgr_cls.return_value.get_status.return_value = (
                self._make_codex_status()
            )
            issues = _validate_auth_for_run(settings)
        # Should not report missing google agent key
        assert not any("google" in i.lower() and "agent" in i.lower() for i in issues)

    def test_openai_agent_provider_missing_key_reports_openai_issue(self):
        settings = self._make_settings(
            agent_provider="openai",
            openai_api_key="",
            cu_provider="openai",
        )
        with patch("haindy.main.OpenAIAuthManager") as mock_mgr_cls:
            mock_mgr_cls.return_value.get_status.return_value = (
                self._make_codex_status()
            )
            issues = _validate_auth_for_run(settings)
        assert any("openai" in i.lower() for i in issues)

    def test_openai_agent_provider_with_key_no_agent_issue(self):
        settings = self._make_settings(
            agent_provider="openai",
            openai_api_key="sk-test",
            cu_provider="openai",
        )
        with patch("haindy.main.OpenAIAuthManager") as mock_mgr_cls:
            mock_mgr_cls.return_value.get_status.return_value = (
                self._make_codex_status()
            )
            issues = _validate_auth_for_run(settings)
        # CU issue may exist, but not agent issue
        agent_issues = [
            i for i in issues if "non-cu" in i.lower() or "planning" in i.lower()
        ]
        assert len(agent_issues) == 0


@pytest.mark.asyncio
async def test_provider_list_command_dispatches() -> None:
    with (
        patch("haindy.main.handle_provider_list", return_value=0) as mock_list,
        patch("haindy.main.ensure_settings_skeleton"),
    ):
        result = await async_main(["provider", "list"])
    assert result == 0
    mock_list.assert_called_once()


@pytest.mark.asyncio
async def test_provider_set_command_dispatches() -> None:
    with (
        patch("haindy.main.handle_provider_set", return_value=0) as mock_set,
        patch("haindy.main.ensure_settings_skeleton"),
    ):
        result = await async_main(["provider", "set", "anthropic"])
    assert result == 0
    mock_set.assert_called_once_with("anthropic")


@pytest.mark.asyncio
async def test_provider_set_computer_use_command_dispatches() -> None:
    with (
        patch(
            "haindy.main.handle_provider_set_computer_use", return_value=0
        ) as mock_cu,
        patch("haindy.main.ensure_settings_skeleton"),
    ):
        result = await async_main(["provider", "set-computer-use", "google"])
    assert result == 0
    mock_cu.assert_called_once_with("google")


@pytest.mark.asyncio
async def test_provider_set_model_command_dispatches() -> None:
    with (
        patch(
            "haindy.main.handle_provider_set_model", return_value=0
        ) as mock_set_model,
        patch("haindy.main.ensure_settings_skeleton"),
    ):
        result = await async_main(
            ["provider", "set-model", "google", "gemini-3-flash-preview"]
        )
    assert result == 0
    mock_set_model.assert_called_once_with(
        "google",
        "gemini-3-flash-preview",
        computer_use=False,
    )
