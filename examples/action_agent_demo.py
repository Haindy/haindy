#!/usr/bin/env python3
"""
Demonstration of the Action Agent.

This script shows how the Action Agent analyzes screenshots with grid overlays
to determine precise coordinates for browser interactions.
"""

import asyncio
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.action_agent import ActionAgent
from src.config.settings import get_settings
from src.core.types import ActionInstruction, ActionType
from src.grid.overlay import GridOverlay
from src.monitoring.logger import setup_logging


def create_demo_screenshot(width: int = 1920, height: int = 1080) -> bytes:
    """Create a demo screenshot with UI elements."""
    # Create a white background
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Draw a simple login form
    # Title
    draw.text((860, 200), "Login", fill='black', font=title_font)
    
    # Username field
    draw.rectangle((760, 300, 1160, 340), outline='gray', width=2)
    draw.text((770, 310), "Username", fill='gray', font=font)
    
    # Password field
    draw.rectangle((760, 380, 1160, 420), outline='gray', width=2)
    draw.text((770, 390), "Password", fill='gray', font=font)
    
    # Login button
    draw.rectangle((860, 480, 1060, 530), fill='#4CAF50', outline='#45a049', width=2)
    draw.text((920, 495), "Login", fill='white', font=font)
    
    # Forgot password link
    draw.text((880, 560), "Forgot Password?", fill='#2196F3', font=font)
    
    # Add some navigation elements
    # Header
    draw.rectangle((0, 0, width, 80), fill='#333333')
    draw.text((50, 25), "HAINDY Demo Application", fill='white', font=title_font)
    
    # Navigation menu
    menu_items = ["Home", "About", "Services", "Contact"]
    x_offset = 500
    for item in menu_items:
        draw.text((x_offset, 30), item, fill='white', font=font)
        x_offset += 120
    
    # Add a sidebar
    draw.rectangle((0, 80, 200, height), fill='#f0f0f0')
    sidebar_items = ["Dashboard", "Profile", "Settings", "Help", "Logout"]
    y_offset = 120
    for item in sidebar_items:
        draw.text((20, y_offset), item, fill='black', font=font)
        y_offset += 50
    
    # Convert to bytes
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def save_demo_images(
    screenshot: bytes, 
    overlay_screenshot: bytes,
    output_dir: Path
) -> None:
    """Save demo images for visualization."""
    output_dir.mkdir(exist_ok=True)
    
    # Save original screenshot
    with open(output_dir / "demo_screenshot.png", "wb") as f:
        f.write(screenshot)
    
    # Save screenshot with grid overlay
    with open(output_dir / "demo_screenshot_grid.png", "wb") as f:
        f.write(overlay_screenshot)
    
    print(f"\nDemo images saved to: {output_dir}")
    print(f"  - Original: demo_screenshot.png")
    print(f"  - With grid: demo_screenshot_grid.png")


async def demonstrate_action_agent():
    """Run Action Agent demonstration."""
    print("HAINDY Action Agent Demonstration")
    print("=" * 50)
    
    # Initialize the agent
    print("\n1. Initializing Action Agent...")
    agent = ActionAgent()
    
    # For demo purposes, we'll mock the AI responses
    async def mock_ai_call(messages, **kwargs):
        """Mock AI responses for demo."""
        # Extract the instruction from the prompt
        prompt = messages[0]["content"][0]["text"]
        
        if "Login button" in prompt:
            return {
                "content": {
                    "cell": "AE14",
                    "offset_x": 0.5,
                    "offset_y": 0.5,
                    "confidence": 0.92,
                    "reasoning": "Login button is clearly visible in the center of cell AE14"
                }
            }
        elif "Username" in prompt:
            return {
                "content": {
                    "cell": "AE9",
                    "offset_x": 0.5,
                    "offset_y": 0.5,
                    "confidence": 0.88,
                    "reasoning": "Username field located in cell AE9"
                }
            }
        elif "Forgot Password" in prompt:
            return {
                "content": {
                    "cell": "AE17",
                    "offset_x": 0.5,
                    "offset_y": 0.5,
                    "confidence": 0.75,  # Lower confidence to trigger refinement
                    "reasoning": "Forgot password link possibly in cell AE17"
                }
            }
        elif "refined" in prompt:
            # Refinement response
            return {
                "content": {
                    "refined_x": 5,
                    "refined_y": 6,
                    "confidence": 0.95,
                    "reasoning": "Link precisely located in refined grid"
                }
            }
        else:
            return {
                "content": {
                    "cell": "A1",
                    "offset_x": 0.5,
                    "offset_y": 0.5,
                    "confidence": 0.5,
                    "reasoning": "Default response"
                }
            }
    
    # Replace the agent's AI call with our mock
    agent.call_ai = mock_ai_call
    
    print("   ✓ Agent initialized with mocked AI responses")
    
    # Create demo screenshot
    print("\n2. Creating demo screenshot...")
    screenshot = create_demo_screenshot()
    print("   ✓ Demo screenshot created (1920x1080)")
    
    # Create grid overlay for visualization
    grid_overlay = GridOverlay(grid_size=60)
    grid_overlay.initialize(1920, 1080)
    # GridOverlay expects bytes, not PIL Image
    overlay_screenshot = grid_overlay.create_overlay_image(screenshot)
    
    # Save demo images
    output_dir = Path("demo_output")
    save_demo_images(screenshot, overlay_screenshot, output_dir)
    
    # Test Case 1: High confidence action (no refinement)
    print("\n3. Test Case 1: Click Login Button (High Confidence)")
    print("   Creating action instruction...")
    
    login_instruction = ActionInstruction(
        action_type=ActionType.CLICK,
        description="Click the login button",
        target="Login button",
        expected_outcome="Login form is submitted"
    )
    
    print("   Analyzing screenshot to find login button...")
    login_action = await agent.determine_action(screenshot, login_instruction)
    
    print(f"\n   ✓ Action determined:")
    print(f"     Cell: {login_action.coordinate.cell}")
    print(f"     Offset: ({login_action.coordinate.offset_x:.2f}, {login_action.coordinate.offset_y:.2f})")
    print(f"     Confidence: {login_action.coordinate.confidence:.2%}")
    print(f"     Refined: {login_action.coordinate.refined}")
    
    # Calculate pixel coordinates
    pixels = grid_overlay.coordinate_to_pixels(login_action.coordinate)
    print(f"     Pixel coordinates: ({pixels[0]}, {pixels[1]})")
    
    # Test Case 2: Medium confidence action (no refinement)
    print("\n4. Test Case 2: Type in Username Field")
    username_instruction = ActionInstruction(
        action_type=ActionType.TYPE,
        description="Type username in the username field",
        target="Username input field",
        value="demo_user",
        expected_outcome="Username is entered"
    )
    
    print("   Analyzing screenshot to find username field...")
    username_action = await agent.determine_action(screenshot, username_instruction)
    
    print(f"\n   ✓ Action determined:")
    print(f"     Cell: {username_action.coordinate.cell}")
    print(f"     Confidence: {username_action.coordinate.confidence:.2%}")
    print(f"     Value to type: {username_action.instruction.value}")
    
    # Test Case 3: Low confidence action (triggers refinement)
    print("\n5. Test Case 3: Click Forgot Password Link (Low Confidence → Refinement)")
    forgot_instruction = ActionInstruction(
        action_type=ActionType.CLICK,
        description="Click the forgot password link",
        target="Forgot Password link",
        expected_outcome="Password recovery page opens"
    )
    
    print("   Analyzing screenshot to find forgot password link...")
    print("   Initial confidence expected to be low, triggering refinement...")
    forgot_action = await agent.determine_action(screenshot, forgot_instruction)
    
    print(f"\n   ✓ Action determined after refinement:")
    print(f"     Cell: {forgot_action.coordinate.cell}")
    print(f"     Offset: ({forgot_action.coordinate.offset_x:.2f}, {forgot_action.coordinate.offset_y:.2f})")
    print(f"     Confidence: {forgot_action.coordinate.confidence:.2%}")
    print(f"     Refined: {forgot_action.coordinate.refined}")
    
    # Test Case 4: Navigation action
    print("\n6. Test Case 4: Navigate to URL")
    nav_instruction = ActionInstruction(
        action_type=ActionType.NAVIGATE,
        description="Navigate to the homepage",
        target="https://example.com",
        expected_outcome="Homepage loads"
    )
    
    print("   Processing navigation instruction...")
    nav_action = await agent.determine_action(screenshot, nav_instruction)
    
    print(f"\n   ✓ Navigation action created")
    print(f"     Target URL: {nav_instruction.target}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print(f"- Action Agent successfully analyzed {4} different actions")
    print(f"- Adaptive refinement triggered when confidence < {agent.confidence_threshold:.0%}")
    print(f"- Grid system using {agent.grid_overlay.grid_size}x{agent.grid_overlay.grid_size} overlay")
    print(f"- All coordinates mapped to precise pixel locations")
    
    print("\nKey Features Demonstrated:")
    print("✓ Screenshot analysis with grid overlay")
    print("✓ Confidence-based coordinate determination")
    print("✓ Adaptive refinement for low-confidence targets")
    print("✓ Support for multiple action types (click, type, navigate)")
    print("✓ Precise pixel coordinate mapping")
    
    print("\nDemo complete!")


async def main():
    """Run the demonstration."""
    # Setup logging
    setup_logging()
    
    # Note: This demo uses mocked AI responses
    print("\nNote: This demo uses mocked AI responses for demonstration purposes.")
    print("In production, the Action Agent would use real OpenAI API calls.")
    print("To run with real AI, set OPENAI_API_KEY environment variable.\n")
    
    try:
        await demonstrate_action_agent()
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())