"""
HAINDY - Autonomous AI Testing Agent
Main entry point for the application.
"""

import asyncio
import sys
from typing import Optional

from src.config.settings import get_settings
from src.monitoring.logger import setup_logging


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for HAINDY.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Initialize configuration
    settings = get_settings()
    
    # Set up logging
    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file,
    )
    
    print("HAINDY - Autonomous AI Testing Agent")
    print("Version 0.1.0")
    print(f"\nConfiguration loaded:")
    print(f"  - OpenAI Model: {settings.openai_model}")
    print(f"  - Grid Size: {settings.grid_size}x{settings.grid_size}")
    print(f"  - Browser Mode: {'Headless' if settings.browser_headless else 'Headed'}")
    print(f"  - Debug Mode: {'Enabled' if settings.debug_mode else 'Disabled'}")
    print("\nCore foundation ready. Awaiting agent implementations.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())