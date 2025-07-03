"""
HAINDY - Autonomous AI Testing Agent
Main entry point for the application.
"""

import sys
from typing import Optional


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for HAINDY.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    print("HAINDY - Autonomous AI Testing Agent")
    print("Version 0.1.0")
    print("\nProject setup complete. Ready for Phase 1 implementation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())