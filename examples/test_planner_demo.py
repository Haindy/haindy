#!/usr/bin/env python3
"""
Demonstration of the Test Planner Agent.

This script shows how the Test Planner Agent creates structured test plans
from high-level requirements.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.test_planner import TestPlannerAgent
from src.config.settings import get_settings
from src.monitoring.logger import setup_logging


def print_test_plan(test_plan):
    """Pretty print a test plan."""
    print(f"\n{'='*60}")
    print(f"Test Plan: {test_plan.name}")
    print(f"{'='*60}")
    print(f"Description: {test_plan.description}")
    print(f"Requirements: {test_plan.requirements[:80]}..." if len(test_plan.requirements) > 80 else f"Requirements: {test_plan.requirements}")
    
    if test_plan.tags:
        print(f"Tags: {', '.join(test_plan.tags)}")
    
    if test_plan.estimated_duration_seconds:
        print(f"Estimated Duration: {test_plan.estimated_duration_seconds} seconds")
    
    print(f"\nSteps ({len(test_plan.steps)}):")
    for step in test_plan.steps:
        print(f"\n  Step {step.step_number}: {step.description}")
        action = step.action_instruction
        print(f"    Action: {action.action_type.value} - {action.target}")
        if action.value:
            print(f"    Value: {action.value}")
        print(f"    Expected: {action.expected_outcome}")
        if step.dependencies:
            print(f"    Dependencies: {len(step.dependencies)} step(s)")
        print(f"    Optional: {'Yes' if step.optional else 'No'}")
        print(f"    Max Retries: {step.max_retries}")
    
    print(f"\n{'='*60}\n")


def main():
    """Run Test Planner Agent demonstration."""
    # Setup logging
    setup_logging()
    
    print("HAINDY Test Planner Agent Demonstration")
    print("=" * 50)
    
    # Initialize the agent
    print("\n1. Initializing Test Planner Agent...")
    agent = TestPlannerAgent()
    
    # Initialize the OpenAI client for the demo
    import openai
    from openai import OpenAI
    openai_client = OpenAI(api_key=settings.openai_api_key)
    agent._client = openai_client
    
    print("   ✓ Agent initialized")
    
    # Example 1: E-commerce checkout flow
    print("\n2. Creating test plan for e-commerce checkout...")
    ecommerce_requirements = """
    Test the complete checkout flow for an e-commerce website:
    - User should be able to add items to cart
    - User should be able to review cart contents
    - User should be able to enter shipping information
    - User should be able to select payment method
    - User should be able to complete the purchase
    - Order confirmation should be displayed
    
    The test should cover both guest checkout and logged-in user checkout.
    """
    
    try:
        ecommerce_plan = agent.create_test_plan(
            ecommerce_requirements,
            context={
                "application": "E-commerce Platform",
                "test_environment": "Staging",
                "user_types": "Guest and Registered Users"
            }
        )
        print_test_plan(ecommerce_plan)
    except Exception as e:
        print(f"   ✗ Error creating e-commerce test plan: {e}")
    
    # Example 2: User registration flow
    print("\n3. Creating test plan for user registration...")
    registration_requirements = """
    Test the user registration process:
    - User should be able to access registration form
    - Form should validate email format
    - Password should meet security requirements
    - User should receive confirmation email
    - User should be able to activate account
    """
    
    try:
        registration_plan = agent.create_test_plan(registration_requirements)
        print_test_plan(registration_plan)
    except Exception as e:
        print(f"   ✗ Error creating registration test plan: {e}")
    
    # Example 3: Extract multiple scenarios
    print("\n4. Extracting test scenarios from complex requirements...")
    complex_requirements = """
    Test a social media posting feature that allows users to:
    - Create text posts with optional images
    - Tag other users in posts
    - Set privacy levels (public, friends, private)
    - Schedule posts for future publishing
    - Edit or delete existing posts
    - React to other users' posts
    - Report inappropriate content
    
    Consider mobile and desktop versions, different user roles (regular user, verified user, admin),
    and edge cases like network failures and content moderation.
    """
    
    try:
        scenarios = agent.extract_test_scenarios(complex_requirements)
        print(f"\nExtracted {len(scenarios)} test scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n  Scenario {i}: {scenario.get('name', 'Unnamed')}")
            print(f"    Description: {scenario.get('description', 'No description')}")
            print(f"    Priority: {scenario.get('priority', 'Not specified')}")
            print(f"    Type: {scenario.get('type', 'Not specified')}")
    except Exception as e:
        print(f"   ✗ Error extracting scenarios: {e}")
    
    # Example 4: Refine a test plan
    if 'ecommerce_plan' in locals():
        print("\n5. Refining test plan based on feedback...")
        feedback = """
        Please add more detail for payment processing steps, including:
        - Credit card validation
        - Handling of declined payments
        - Support for multiple payment methods (credit card, PayPal, etc.)
        - Security considerations for payment data
        """
        
        try:
            refined_plan = agent.refine_test_plan(ecommerce_plan, feedback)
            print("\nRefined test plan created:")
            print(f"  Original steps: {len(ecommerce_plan.steps)}")
            print(f"  Refined steps: {len(refined_plan.steps)}")
            print(f"  New plan name: {refined_plan.name}")
        except Exception as e:
            print(f"   ✗ Error refining test plan: {e}")
    
    print("\nDemo complete!")
    
    # Save example output
    if 'ecommerce_plan' in locals():
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save test plan as JSON
        plan_dict = {
            "name": ecommerce_plan.name,
            "description": ecommerce_plan.description,
            "requirements": ecommerce_plan.requirements,
            "tags": ecommerce_plan.tags,
            "estimated_duration_seconds": ecommerce_plan.estimated_duration_seconds,
            "steps": [
                {
                    "step_number": step.step_number,
                    "description": step.description,
                    "action": step.action_instruction.action_type.value,
                    "target": step.action_instruction.target,
                    "value": step.action_instruction.value,
                    "expected_result": step.action_instruction.expected_result,
                    "dependencies": [str(dep_id) for dep_id in step.dependencies],
                    "optional": step.optional,
                    "max_retries": step.max_retries
                }
                for step in ecommerce_plan.steps
            ]
        }
        
        with open(output_dir / "sample_test_plan.json", "w") as f:
            json.dump(plan_dict, f, indent=2)
        
        print(f"\nTest plan saved to: {output_dir / 'sample_test_plan.json'}")


if __name__ == "__main__":
    # Check for OpenAI API key
    settings = get_settings()
    if not settings.openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it to run this demo")
        sys.exit(1)
    
    main()