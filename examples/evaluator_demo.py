#!/usr/bin/env python3
"""
Demonstration of the Evaluator Agent.

This script shows how the Evaluator Agent analyzes screenshots to determine
if test outcomes match expectations.
"""

import asyncio
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.evaluator import EvaluatorAgent
from src.config.settings import get_settings
from src.monitoring.logger import setup_logging


def create_success_screenshot(width: int = 1920, height: int = 1080) -> bytes:
    """Create a screenshot showing successful login."""
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a better font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Draw dashboard header
    draw.rectangle((0, 0, width, 80), fill='#2196F3')
    draw.text((50, 25), "Dashboard - Welcome back!", fill='white', font=title_font)
    
    # Success message
    draw.rectangle((700, 120, 1220, 180), fill='#4CAF50')
    draw.text((860, 140), "✓ Login successful", fill='white', font=font)
    
    # Dashboard content
    draw.text((100, 220), "Recent Activity", fill='black', font=title_font)
    draw.rectangle((100, 260, 600, 500), outline='#ddd', width=2)
    
    draw.text((700, 220), "Statistics", fill='black', font=title_font)
    draw.rectangle((700, 260, 1200, 500), outline='#ddd', width=2)
    
    # Add some data
    draw.text((120, 280), "• User logged in at 10:30 AM", fill='black', font=font)
    draw.text((120, 310), "• 5 new messages", fill='black', font=font)
    draw.text((120, 340), "• 2 pending tasks", fill='black', font=font)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def create_error_screenshot(width: int = 1920, height: int = 1080) -> bytes:
    """Create a screenshot showing login error."""
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Header
    draw.rectangle((0, 0, width, 80), fill='#333333')
    draw.text((50, 25), "Login Page", fill='white', font=title_font)
    
    # Error message
    draw.rectangle((600, 200, 1320, 280), fill='#f44336')
    draw.text((760, 220), "⚠ Invalid username or password", fill='white', font=title_font)
    draw.text((820, 250), "Please try again", fill='white', font=font)
    
    # Login form (still visible)
    draw.text((760, 320), "Username", fill='gray', font=font)
    draw.rectangle((760, 350, 1160, 390), outline='#f44336', width=2)  # Red border
    
    draw.text((760, 420), "Password", fill='gray', font=font)
    draw.rectangle((760, 450, 1160, 490), outline='#f44336', width=2)  # Red border
    
    # Login button
    draw.rectangle((860, 540, 1060, 590), fill='#4CAF50', outline='#45a049', width=2)
    draw.text((920, 555), "Login", fill='white', font=font)
    
    # Error icon
    draw.ellipse((550, 210, 590, 250), fill='#f44336', outline='white', width=3)
    draw.text((565, 220), "!", fill='white', font=title_font)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def create_partial_success_screenshot(width: int = 1920, height: int = 1080) -> bytes:
    """Create a screenshot showing partial success (dashboard with some errors)."""
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Dashboard header
    draw.rectangle((0, 0, width, 80), fill='#2196F3')
    draw.text((50, 25), "Dashboard", fill='white', font=title_font)
    
    # Warning message
    draw.rectangle((700, 120, 1220, 180), fill='#FF9800')
    draw.text((780, 140), "⚠ Some widgets failed to load", fill='white', font=font)
    
    # Dashboard content - some loaded, some with errors
    draw.text((100, 220), "Recent Activity", fill='black', font=title_font)
    draw.rectangle((100, 260, 600, 500), outline='#ddd', width=2)
    draw.text((120, 280), "• User logged in", fill='black', font=font)
    
    # Statistics section with error
    draw.text((700, 220), "Statistics", fill='black', font=title_font)
    draw.rectangle((700, 260, 1200, 500), fill='#ffebee', outline='#f44336', width=2)
    draw.text((850, 360), "Failed to load data", fill='#f44336', font=font)
    draw.text((870, 390), "Error code: 503", fill='#999', font=font)
    
    # Loading spinner placeholder
    draw.text((1300, 220), "Notifications", fill='black', font=title_font)
    draw.rectangle((1300, 260, 1800, 500), outline='#ddd', width=2)
    draw.text((1450, 360), "Loading...", fill='#999', font=font)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


async def demonstrate_evaluator():
    """Run Evaluator Agent demonstration."""
    print("HAINDY Evaluator Agent Demonstration")
    print("=" * 50)
    
    # Initialize the agent
    print("\n1. Initializing Evaluator Agent...")
    agent = EvaluatorAgent()
    
    # For demo purposes, mock the AI responses
    async def mock_ai_success(messages, **kwargs):
        return {
            "content": {
                "success": True,
                "confidence": 0.95,
                "actual_outcome": "Dashboard successfully loaded with welcome message and user data visible",
                "deviations": [],
                "suggestions": [],
                "detailed_analysis": {
                    "elements_found": ["Dashboard header", "Welcome message", "Success notification", "Recent activity", "Statistics panel"],
                    "text_content": ["Dashboard - Welcome back!", "Login successful", "Recent Activity", "User logged in at 10:30 AM"],
                    "ui_state": "Dashboard fully loaded and functional",
                    "error_indicators": [],
                    "success_indicators": ["Green success banner", "Welcome message", "User data displayed"]
                }
            }
        }
    
    async def mock_ai_error(messages, **kwargs):
        return {
            "content": {
                "success": False,
                "confidence": 0.92,
                "actual_outcome": "Login page still displayed with error message showing invalid credentials",
                "deviations": [
                    "Expected dashboard but still on login page",
                    "Error message displayed: 'Invalid username or password'",
                    "Form fields have red borders indicating validation error"
                ],
                "suggestions": [
                    "Verify correct credentials are being used",
                    "Check if user account exists and is active",
                    "Ensure password is entered correctly",
                    "Look for any account lockout or rate limiting"
                ],
                "detailed_analysis": {
                    "elements_found": ["Login form", "Error message", "Username field", "Password field", "Error icon"],
                    "text_content": ["Invalid username or password", "Please try again", "Login"],
                    "ui_state": "Login page with error state",
                    "error_indicators": ["Red error banner", "Red input borders", "Error icon"],
                    "success_indicators": []
                }
            }
        }
    
    async def mock_ai_partial(messages, **kwargs):
        return {
            "content": {
                "success": True,
                "confidence": 0.7,
                "actual_outcome": "Dashboard loaded but some widgets failed to load properly",
                "deviations": [
                    "Statistics widget showing error instead of data",
                    "Notifications panel still loading",
                    "Warning banner about failed widgets"
                ],
                "suggestions": [
                    "Wait longer for all widgets to load",
                    "Check API endpoints for statistics service",
                    "Investigate error code 503 (Service Unavailable)",
                    "Consider retry mechanism for failed widgets"
                ],
                "detailed_analysis": {
                    "elements_found": ["Dashboard header", "Warning message", "Recent activity", "Error in statistics", "Loading spinner"],
                    "text_content": ["Dashboard", "Some widgets failed to load", "Failed to load data", "Error code: 503"],
                    "ui_state": "Dashboard partially loaded with some failures",
                    "error_indicators": ["Orange warning banner", "Red bordered statistics widget", "Error code 503"],
                    "success_indicators": ["Dashboard loaded", "Recent activity visible"]
                }
            }
        }
    
    print("   ✓ Agent initialized with mocked AI responses")
    
    # Test Case 1: Successful login evaluation
    print("\n2. Test Case 1: Successful Login")
    print("   Expected: User should be redirected to dashboard after login")
    
    success_screenshot = create_success_screenshot()
    agent.call_ai = mock_ai_success
    
    result = await agent.evaluate_result(
        success_screenshot,
        "User should be redirected to dashboard with welcome message after successful login"
    )
    
    print(f"\n   Evaluation Result:")
    print(f"   ✓ Success: {result.success}")
    print(f"   ✓ Confidence: {result.confidence:.0%}")
    print(f"   ✓ Outcome: {result.actual_outcome}")
    print(f"   ✓ Deviations: {len(result.deviations)} found")
    print(f"   ✓ Success Indicators: {len(result.screenshot_analysis['success_indicators'])} found")
    
    # Test Case 2: Failed login evaluation
    print("\n3. Test Case 2: Failed Login")
    print("   Expected: User should be redirected to dashboard after login")
    
    error_screenshot = create_error_screenshot()
    agent.call_ai = mock_ai_error
    
    result = await agent.evaluate_result(
        error_screenshot,
        "User should be redirected to dashboard after login"
    )
    
    print(f"\n   Evaluation Result:")
    print(f"   ✗ Success: {result.success}")
    print(f"   ✓ Confidence: {result.confidence:.0%}")
    print(f"   ✓ Outcome: {result.actual_outcome}")
    print(f"   ✓ Deviations: {len(result.deviations)} found")
    for i, deviation in enumerate(result.deviations, 1):
        print(f"      {i}. {deviation}")
    print(f"   ✓ Suggestions: {len(result.suggestions)} provided")
    for i, suggestion in enumerate(result.suggestions[:2], 1):
        print(f"      {i}. {suggestion}")
    
    # Test Case 3: Partial success evaluation
    print("\n4. Test Case 3: Partial Success")
    print("   Expected: Dashboard should load with all widgets functioning")
    
    partial_screenshot = create_partial_success_screenshot()
    agent.call_ai = mock_ai_partial
    
    result = await agent.evaluate_result(
        partial_screenshot,
        "Dashboard should load completely with all widgets functioning"
    )
    
    print(f"\n   Evaluation Result:")
    print(f"   ~ Success: {result.success} (Partial)")
    print(f"   ✓ Confidence: {result.confidence:.0%} (Lower confidence indicates issues)")
    print(f"   ✓ Outcome: {result.actual_outcome}")
    print(f"   ✓ Deviations: {len(result.deviations)} found")
    for deviation in result.deviations:
        print(f"      - {deviation}")
    
    # Test Case 4: Error detection
    print("\n5. Test Case 4: Error Detection")
    print("   Checking error screenshot for specific error indicators...")
    
    async def mock_ai_error_check(messages, **kwargs):
        return {
            "content": {
                "has_errors": True,
                "error_count": 3,
                "errors": [
                    {
                        "type": "validation",
                        "message": "Invalid username or password",
                        "location": "Top center of screen in red banner",
                        "severity": "high"
                    },
                    {
                        "type": "ui",
                        "message": "Form fields have error styling",
                        "location": "Username and password input fields",
                        "severity": "medium"
                    },
                    {
                        "type": "authentication",
                        "message": "Login attempt failed",
                        "location": "Login form",
                        "severity": "high"
                    }
                ],
                "confidence": 0.95
            }
        }
    
    agent.call_ai = mock_ai_error_check
    error_analysis = await agent.check_for_errors(error_screenshot)
    
    print(f"\n   Error Analysis:")
    print(f"   ✓ Errors Detected: {error_analysis['has_errors']}")
    print(f"   ✓ Error Count: {error_analysis['error_count']}")
    print(f"   ✓ Confidence: {error_analysis['confidence']:.0%}")
    for error in error_analysis['errors']:
        print(f"      - {error['type'].upper()}: {error['message']} (Severity: {error['severity']})")
    
    # Summary
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print("- Evaluated 3 different UI states (success, failure, partial)")
    print("- Detected specific errors and deviations")
    print("- Provided actionable suggestions for failures")
    print("- Demonstrated confidence-based evaluation")
    print("- Showed specialized error detection capabilities")
    
    print("\nKey Features Demonstrated:")
    print("✓ Screenshot analysis for outcome verification")
    print("✓ Success/failure detection with confidence scores")
    print("✓ Detailed deviation identification")
    print("✓ Actionable suggestions for test failures")
    print("✓ Specialized error detection and classification")
    
    print("\nDemo complete!")


async def main():
    """Run the demonstration."""
    # Setup logging
    setup_logging()
    
    print("\nNote: This demo uses mocked AI responses for demonstration purposes.")
    print("In production, the Evaluator Agent would use real OpenAI API calls.")
    print("To run with real AI, set OPENAI_API_KEY environment variable.\n")
    
    try:
        await demonstrate_evaluator()
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())