"""Tests for main.py CLI interface."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.main import (
    async_main,
    create_parser,
    get_interactive_requirements,
    read_plan_file,
    show_version,
    test_api_connection,
)


class TestCLIParser:
    """Test command line parser."""
    
    def test_parser_creation(self):
        """Test parser is created with all expected arguments."""
        parser = create_parser()
        
        # Check description
        assert "HAINDY - Autonomous AI Testing Agent v0.1.0" in parser.description
        
        # Get all actions
        actions = {action.dest: action for action in parser._actions}
        
        # Check all new arguments exist
        assert "requirements" in actions
        assert "plan" in actions
        assert "json_test_plan" in actions
        assert "test_api" in actions
        assert "version" in actions
        assert "berserk" in actions
    
    def test_requirements_is_action(self):
        """Test that --requirements is now an action, not taking a value."""
        parser = create_parser()
        args = parser.parse_args(["--requirements"])
        assert args.requirements is True
    
    def test_mutually_exclusive_inputs(self):
        """Test that input options are mutually exclusive."""
        parser = create_parser()
        
        # These should work
        parser.parse_args(["--requirements"])
        parser.parse_args(["--plan", "test.md"])
        parser.parse_args(["--json-test-plan", "test.json"])
        parser.parse_args(["--test-api"])
        parser.parse_args(["--version"])
        
        # These should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["--requirements", "--plan", "test.md"])
        
        with pytest.raises(SystemExit):
            parser.parse_args(["--plan", "test.md", "--json-test-plan", "test.json"])


class TestUtilityCommands:
    """Test utility command functions."""
    
    def test_show_version(self, capsys):
        """Test version display."""
        result = show_version()
        captured = capsys.readouterr()
        
        assert result == 0
        assert "HAINDY - Autonomous AI Testing Agent" in captured.out
        assert "Version: 0.1.0" in captured.out
        assert "Python: 3.10+" in captured.out
    
    @pytest.mark.asyncio
    async def test_api_connection_success(self, capsys):
        """Test successful API connection test."""
        mock_response = {
            "content": "API test successful",
            "model": "gpt-5",
            "usage": {"total_tokens": 10}
        }
        
        with patch("src.models.openai_client.OpenAIClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await test_api_connection()
            captured = capsys.readouterr()
            
            assert result == 0
            assert "OpenAI API connection successful" in captured.out
            assert "Model: gpt-5" in captured.out
    
    @pytest.mark.asyncio
    async def test_api_connection_failure(self, capsys):
        """Test failed API connection test."""
        with patch("src.models.openai_client.OpenAIClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call = AsyncMock(side_effect=Exception("API Error"))
            mock_client_class.return_value = mock_client
            
            result = await test_api_connection()
            captured = capsys.readouterr()
            
            assert result == 1
            assert "API test failed" in captured.out
            assert "OPENAI_API_KEY" in captured.out


class TestInteractiveMode:
    """Test interactive requirements mode."""
    
    def test_get_interactive_requirements(self, monkeypatch, capsys):
        """Test interactive requirements input."""
        # Mock user input
        inputs = iter([
            "Test the login flow",
            "Enter username and password",
            "Click login button",
            "",  # First empty line
            "",  # Second empty line triggers end
        ])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))
        
        # Mock URL prompt
        with patch("src.main.Prompt.ask", return_value="https://example.com"):
            requirements, url = get_interactive_requirements()
        
        assert "Test the login flow" in requirements
        assert "Enter username and password" in requirements
        assert url == "https://example.com"
        
        captured = capsys.readouterr()
        assert "HAINDY Interactive Mode" in captured.out
    
    def test_get_interactive_requirements_no_input(self, monkeypatch):
        """Test interactive mode with no input."""
        # Mock empty input
        inputs = iter(["", ""])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))
        
        with pytest.raises(SystemExit):
            get_interactive_requirements()


class TestPlanFileReading:
    """Test plan file reading."""
    
    @pytest.mark.asyncio
    async def test_read_plan_file_not_found(self, capsys):
        """Test reading non-existent file."""
        with pytest.raises(SystemExit):
            await read_plan_file(Path("nonexistent.md"))
        captured = capsys.readouterr()
        assert "File not found" in captured.out
    
    @pytest.mark.asyncio
    async def test_read_plan_file_with_url(self, tmp_path):
        """Test reading plan file with URL on first line."""
        # Create test file with URL
        test_file = tmp_path / "test_requirements.md"
        test_file.write_text("https://example.com\nTest the login flow\nVerify user can log in")
        
        requirements, url = await read_plan_file(test_file)
        
        assert url == "https://example.com"
        assert "Test the login flow" in requirements
        assert "Verify user can log in" in requirements
        assert "https://example.com" not in requirements  # URL should be removed from requirements
    
    @pytest.mark.asyncio
    async def test_read_plan_file_without_url(self, tmp_path):
        """Test reading plan file without URL - should prompt."""
        test_file = tmp_path / "test.md"
        test_file.write_text("Test the shopping cart functionality")
        
        # Mock user input for URL
        with patch("src.main.Prompt.ask", return_value="https://shop.example.com"):
            requirements, url = await read_plan_file(test_file)
        
        assert url == "https://shop.example.com"
        assert requirements == "Test the shopping cart functionality"


class TestMainFlow:
    """Test main execution flow."""
    
    @pytest.mark.asyncio
    async def test_version_command(self):
        """Test --version command."""
        with patch("src.main.show_version", return_value=0) as mock_version:
            result = await async_main(["--version"])
            assert result == 0
            mock_version.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_api_command(self):
        """Test --test-api command."""
        with patch("src.main.test_api_connection", new_callable=AsyncMock) as mock_test:
            mock_test.return_value = 0
            result = await async_main(["--test-api"])
            assert result == 0
            mock_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_requirements_mode(self):
        """Test --requirements interactive mode."""
        with patch("src.main.get_interactive_requirements") as mock_get_req:
            mock_get_req.return_value = ("test requirements", "https://example.com")
            
            with patch("src.main.run_test", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = 0
                result = await async_main(["--requirements"])
                
                assert result == 0
                mock_get_req.assert_called_once()
                mock_run.assert_called_once()
                
                # Check that requirements and URL were passed correctly
                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["requirements"] == "test requirements"
                assert call_kwargs["url"] == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_plan_mode(self):
        """Test --plan mode."""
        with patch("src.main.read_plan_file", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = ("Test requirements", "https://example.com")
            with patch("src.main.run_test", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = 0
                result = await async_main(["--plan", "test.md"])
                assert result == 0
                mock_read.assert_called_once()
                mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_berserk_mode_flag(self):
        """Test --berserk flag is passed through."""
        with patch("src.main.read_plan_file", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = ("Test requirements", "https://example.com")
            with patch("src.main.run_test", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = 0
                result = await async_main(["--plan", "test.md", "--berserk"])
                assert result == 0
                # Verify berserk flag was passed to run_test
                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["berserk"] is True
    
    @pytest.mark.asyncio
    async def test_json_test_plan_mode(self):
        """Test --json-test-plan mode."""
        test_scenario = {
            "name": "Test",
            "requirements": "Test requirements",
            "url": "https://example.com"
        }
        
        with patch("src.main.load_scenario") as mock_load:
            mock_load.return_value = test_scenario
            
            with patch("src.main.run_test", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = 0
                result = await async_main(["--json-test-plan", "test.json"])
                
                assert result == 0
                mock_load.assert_called_once_with(Path("test.json"))
                mock_run.assert_called_once()
                
                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["requirements"] == "Test requirements"
                assert call_kwargs["url"] == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_no_input_shows_help(self, capsys):
        """Test that no input shows help."""
        result = await async_main([])
        assert result == 1
        
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "HAINDY - Autonomous AI Testing Agent" in captured.out
