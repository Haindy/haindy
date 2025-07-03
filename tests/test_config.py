"""
Unit tests for configuration management.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import ConfigManager, Settings, get_settings


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.openai_temperature == 0.7
        assert settings.grid_size == 60
        assert settings.grid_refinement_enabled is True
        assert settings.browser_headless is True
        assert settings.browser_viewport_width == 1920
        assert settings.browser_viewport_height == 1080
        assert settings.max_test_steps == 100
        assert settings.log_level == "INFO"
        assert settings.debug_mode is False

    def test_settings_from_env(self):
        """Test loading settings from environment variables."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key-123",
            "OPENAI_MODEL": "gpt-4",
            "GRID_SIZE": "80",
            "LOG_LEVEL": "DEBUG",
            "DEBUG_MODE": "true",
        }):
            settings = Settings()
            
            assert settings.openai_api_key == "test-key-123"
            assert settings.openai_model == "gpt-4"
            assert settings.grid_size == 80
            assert settings.log_level == "DEBUG"
            assert settings.debug_mode is True

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid level
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"
        
        # Invalid level
        with pytest.raises(ValueError, match="Invalid log level"):
            Settings(log_level="INVALID")

    def test_log_format_validation(self):
        """Test log format validation."""
        # Valid formats
        settings = Settings(log_format="json")
        assert settings.log_format == "json"
        
        settings = Settings(log_format="text")
        assert settings.log_format == "text"
        
        # Invalid format
        with pytest.raises(ValueError, match="Invalid log format"):
            Settings(log_format="xml")

    def test_numeric_validation(self):
        """Test numeric field validation."""
        # Valid values
        settings = Settings(
            openai_temperature=1.5,
            grid_size=50,
            browser_viewport_width=1024,
        )
        assert settings.openai_temperature == 1.5
        assert settings.grid_size == 50
        assert settings.browser_viewport_width == 1024
        
        # Invalid values
        with pytest.raises(ValueError):
            Settings(openai_temperature=3.0)  # > 2.0
        
        with pytest.raises(ValueError):
            Settings(grid_size=5)  # < 10
        
        with pytest.raises(ValueError):
            Settings(browser_viewport_width=600)  # < 800

    def test_path_fields(self):
        """Test path configuration fields."""
        settings = Settings(
            data_dir=Path("/tmp/data"),
            reports_dir=Path("/tmp/reports"),
        )
        
        assert settings.data_dir == Path("/tmp/data")
        assert settings.reports_dir == Path("/tmp/reports")
        assert isinstance(settings.screenshots_dir, Path)
        assert isinstance(settings.cache_dir, Path)

    def test_create_directories(self, tmp_path):
        """Test directory creation."""
        settings = Settings(
            data_dir=tmp_path / "data",
            reports_dir=tmp_path / "reports",
            screenshots_dir=tmp_path / "screenshots",
            cache_dir=tmp_path / "cache",
        )
        
        # Directories shouldn't exist yet
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "reports").exists()
        
        # Create directories
        settings.create_directories()
        
        # Now they should exist
        assert (tmp_path / "data").exists()
        assert (tmp_path / "reports").exists()
        assert (tmp_path / "screenshots").exists()
        assert (tmp_path / "cache").exists()


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_get_existing_key(self):
        """Test getting an existing configuration key."""
        settings = Settings(grid_size=70)
        config = ConfigManager(settings)
        
        assert config.get("grid_size") == 70
        assert config.get("browser_headless") is True

    def test_get_missing_key(self):
        """Test getting a missing configuration key."""
        settings = Settings()
        config = ConfigManager(settings)
        
        assert config.get("non_existent_key") is None
        assert config.get("non_existent_key", "default") == "default"

    def test_get_required_existing(self):
        """Test getting a required existing key."""
        settings = Settings(openai_model="gpt-4")
        config = ConfigManager(settings)
        
        assert config.get_required("openai_model") == "gpt-4"

    def test_get_required_missing(self):
        """Test getting a required missing key."""
        settings = Settings()
        config = ConfigManager(settings)
        
        with pytest.raises(KeyError, match="Required configuration key not found"):
            config.get_required("non_existent_key")

    def test_get_all(self):
        """Test getting all configuration values."""
        settings = Settings(
            grid_size=75,
            debug_mode=True,
            log_level="DEBUG",
        )
        config = ConfigManager(settings)
        
        all_config = config.get_all()
        
        assert isinstance(all_config, dict)
        assert all_config["grid_size"] == 75
        assert all_config["debug_mode"] is True
        assert all_config["log_level"] == "DEBUG"
        assert "openai_temperature" in all_config
        assert "browser_viewport_width" in all_config


class TestGetSettings:
    """Tests for get_settings function."""

    @patch("src.config.settings.load_dotenv")
    @patch("src.config.settings.Path")
    def test_get_settings_loads_env(self, mock_path_class, mock_load_dotenv, tmp_path):
        """Test that get_settings loads .env file."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")
        
        # Mock Path instance
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance
        
        # Clear the cache
        get_settings.cache_clear()
        
        settings = get_settings()
        
        # Verify load_dotenv was called
        mock_load_dotenv.assert_called_once()

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        # Clear the cache
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same instance
        assert settings1 is settings2