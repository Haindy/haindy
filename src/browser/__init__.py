"""
Browser automation module exports.
"""

from src.browser.controller import BrowserController
from src.browser.driver import PlaywrightDriver

__all__ = [
    "PlaywrightDriver",
    "BrowserController",
]