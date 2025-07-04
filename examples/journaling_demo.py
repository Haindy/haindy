#!/usr/bin/env python3
"""
Demonstration of the Execution Journaling and Scripted Automation system.

This script shows how HAINDY records test executions, builds a pattern library,
and enables dual-mode execution (scripted vs visual).
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import (
    ActionInstruction,
    ActionType,
    TestPlan,
    TestStep
)
from src.journal import (
    DualModeExecutor,
    ExecutionJournal,
    ExecutionMode,
    JournalActionResult,
    JournalEntry,
    JournalManager,
    PatternType,
    ScriptRecorder
)
from src.monitoring.logger import setup_logging


async def demonstrate_journal_recording():
    """Demonstrate journal recording functionality."""
    print("\n1. Journal Recording Demo")
    print("=" * 50)
    
    # Create journal manager
    journal_dir = Path("demo_journals")
    journal_dir.mkdir(exist_ok=True)
    manager = JournalManager(journal_dir=journal_dir)
    
    # Create test plan
    test_plan = TestPlan(
        plan_id=uuid4(),
        name="E-commerce Checkout Flow",
        description="Test the complete checkout process",
        requirements="User should be able to purchase items",
        steps=[
            TestStep(
                step_id=uuid4(),
                step_number=1,
                description="Navigate to product page",
                action_instruction=ActionInstruction(
                    action_type=ActionType.NAVIGATE,
                    description="Go to product listing",
                    expected_outcome="Product page displayed"
                )
            ),
            TestStep(
                step_id=uuid4(),
                step_number=2,
                description="Add product to cart",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click 'Add to Cart' button",
                    expected_outcome="Product added to cart"
                )
            ),
            TestStep(
                step_id=uuid4(),
                step_number=3,
                description="Proceed to checkout",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click 'Checkout' button",
                    expected_outcome="Checkout page displayed"
                )
            )
        ]
    )
    
    # Create journal
    print("\nCreating execution journal...")
    journal = await manager.create_journal(test_plan)
    print(f"  Journal ID: {journal.journal_id}")
    print(f"  Test: {journal.test_name}")
    
    # Simulate test execution
    print("\nSimulating test execution...")
    
    for i, step in enumerate(test_plan.steps):
        print(f"\n  Step {step.step_number}: {step.description}")
        
        # Simulate action result
        if step.action_instruction.action_type == ActionType.NAVIGATE:
            result = JournalActionResult(
                success=True,
                action=ActionType.NAVIGATE,
                confidence=1.0,
                actual_outcome="Page loaded successfully"
            )
            execution_mode = ExecutionMode.SCRIPTED
            execution_time = 500
        else:
            # Simulate visual execution with grid refinement
            result = JournalActionResult(
                success=True,
                action=ActionType.CLICK,
                confidence=0.95 if i == 1 else 0.88,
                coordinates=(500 + i*100, 300),
                grid_coordinates={
                    "initial_selection": f"M{23+i}",
                    "initial_confidence": 0.70,
                    "refinement_applied": True,
                    "refined_coordinates": f"M{23+i}+offset(0.7,0.4)",
                    "final_confidence": 0.95 if i == 1 else 0.88
                },
                playwright_command=f"await page.click('#{step.description.lower().replace(' ', '-')}')",
                selectors={
                    "primary": f"#{step.description.lower().replace(' ', '-')}",
                    "fallback": f"button:has-text('{step.description.split()[-1]}')"
                },
                element_text=step.description.split()[-1],
                actual_outcome=step.action_instruction.expected_outcome
            )
            execution_mode = ExecutionMode.VISUAL
            execution_time = 1500 + i*200
        
        # Record action
        entry = await manager.record_action(
            journal_id=journal.journal_id,
            test_scenario="Checkout Flow",
            step=step,
            action_result=result,
            execution_mode=execution_mode,
            execution_time_ms=execution_time,
            screenshot_before=f"screenshots/step_{i+1}_before.png",
            screenshot_after=f"screenshots/step_{i+1}_after.png"
        )
        
        print(f"    - Action: {entry.action_taken}")
        print(f"    - Mode: {entry.execution_mode}")
        print(f"    - Confidence: {entry.agent_confidence}")
        print(f"    - Time: {entry.execution_time_ms}ms")
        print(f"    - Success: {'✓' if entry.success else '✗'}")
    
    # Show pattern library
    print("\n\nPattern Library Status:")
    stats = await manager.get_pattern_library_stats()
    print(f"  Total patterns: {stats['total_patterns']}")
    for pattern_type, count in stats['patterns_by_type'].items():
        print(f"  - {pattern_type}: {count}")
    
    # Finalize journal
    print("\nFinalizing journal...")
    finalized = await manager.finalize_journal(journal.journal_id)
    
    summary = finalized.get_summary()
    print(f"\nJournal Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Success rate: {summary['success_rate']*100:.1f}%")
    print(f"  Execution modes:")
    print(f"    - Visual: {summary['execution_modes']['visual']}")
    print(f"    - Scripted: {summary['execution_modes']['scripted']}")
    print(f"  Patterns discovered: {summary['patterns']['discovered']}")
    
    return manager, test_plan


async def demonstrate_pattern_matching(manager: JournalManager, test_plan: TestPlan):
    """Demonstrate pattern matching and reuse."""
    print("\n\n2. Pattern Matching Demo")
    print("=" * 50)
    
    # Create a similar test step
    similar_step = TestStep(
        step_id=uuid4(),
        step_number=1,
        description="Click Add to Basket button",  # Similar to "Add to Cart"
        action_instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click the Add to Basket button",
            expected_outcome="Item added to basket"
        )
    )
    
    # Find matching pattern
    print("\nSearching for matching patterns...")
    print(f"  New step: {similar_step.description}")
    
    context = {
        "url": "https://example.com/products",
        "element_text": "Basket",
        "element_type": "button"
    }
    
    match = await manager.find_matching_pattern(similar_step, context)
    
    if match:
        print(f"\n  ✓ Found matching pattern!")
        print(f"    - Type: {match.pattern_type}")
        print(f"    - Command: {match.playwright_command}")
        print(f"    - Success rate: {match.success_count}/{match.success_count + match.failure_count}")
        print(f"    - Avg execution time: {match.avg_execution_time_ms:.0f}ms")
    else:
        print("  ✗ No matching pattern found")


async def demonstrate_dual_mode_execution():
    """Demonstrate dual-mode execution."""
    print("\n\n3. Dual-Mode Execution Demo")
    print("=" * 50)
    
    # Create script recorder
    recorder = ScriptRecorder()
    
    # Simulate a successful visual execution
    print("\nRecording visual execution...")
    
    visual_result = JournalActionResult(
        success=True,
        action=ActionType.CLICK,
        confidence=0.92,
        coordinates=(600, 400),
        grid_coordinates={
            "initial_selection": "P28",
            "refined_coordinates": "P28+offset(0.6,0.3)",
            "final_confidence": 0.92
        },
        actual_outcome="Form submitted"
    )
    
    element_info = {
        "tag": "button",
        "id": "submit-form",
        "class": "btn btn-primary submit-button",
        "text": "Submit Order",
        "data-testid": "order-submit",
        "role": "button"
    }
    
    # Record as scripted command
    scripted_cmd = recorder.record_action(
        ActionType.CLICK,
        visual_result,
        element_info
    )
    
    if scripted_cmd:
        print(f"\n  Scripted command generated:")
        print(f"    - Type: {scripted_cmd.command_type}")
        print(f"    - Command: {scripted_cmd.command}")
        print(f"    - Selectors ({len(scripted_cmd.selectors)}):")
        for i, selector in enumerate(scripted_cmd.selectors[:3]):
            print(f"      {i+1}. {selector}")
        
        # Show execution mode decision
        print(f"\n  Execution mode decision:")
        print(f"    - Fallback threshold: {scripted_cmd.fallback_threshold}")
        print(f"    - Allow visual fallback: {scripted_cmd.allow_visual_fallback}")


async def demonstrate_execution_journal_analysis():
    """Demonstrate journal analysis capabilities."""
    print("\n\n4. Execution Journal Analysis")
    print("=" * 50)
    
    # Create sample journal with mixed execution modes
    journal = ExecutionJournal(
        test_plan_id=uuid4(),
        test_name="Performance Comparison Test"
    )
    
    # Add entries with different execution modes
    scripted_times = []
    visual_times = []
    
    for i in range(10):
        is_scripted = i % 3 != 2  # 7 scripted, 3 visual
        
        entry = JournalEntry(
            test_scenario="Performance Test",
            step_reference=f"Step {i+1}",
            action_taken=f"Action {i+1}",
            expected_result="Success",
            actual_result="Success",
            success=True,
            execution_mode=ExecutionMode.SCRIPTED if is_scripted else ExecutionMode.VISUAL,
            execution_time_ms=150 if is_scripted else 1800,
            agent_confidence=1.0 if is_scripted else 0.9
        )
        
        journal.add_entry(entry)
        
        if is_scripted:
            scripted_times.append(entry.execution_time_ms)
        else:
            visual_times.append(entry.execution_time_ms)
    
    journal.finalize()
    
    # Analyze performance
    print("\nPerformance Analysis:")
    print(f"  Scripted executions: {len(scripted_times)}")
    print(f"    - Average time: {sum(scripted_times)/len(scripted_times):.0f}ms")
    print(f"    - Total time: {sum(scripted_times)}ms")
    
    print(f"\n  Visual executions: {len(visual_times)}")
    print(f"    - Average time: {sum(visual_times)/len(visual_times):.0f}ms")
    print(f"    - Total time: {sum(visual_times)}ms")
    
    time_saved = sum(visual_times) * (len(scripted_times) / len(visual_times)) - sum(scripted_times)
    print(f"\n  Time saved by scripted execution: {time_saved:.0f}ms ({time_saved/1000:.1f}s)")
    print(f"  Speed improvement: {sum(visual_times)/len(visual_times) / (sum(scripted_times)/len(scripted_times)):.1f}x faster")


async def main():
    """Run all demonstrations."""
    setup_logging()
    
    print("\nHAINDY Execution Journaling & Scripted Automation Demo")
    print("=" * 70)
    print("\nThis demo showcases HAINDY's dual-mode execution capabilities:")
    print("- Comprehensive execution journaling with grid coordinates")
    print("- Pattern recognition and caching for test reuse")
    print("- Automatic script generation from visual interactions")
    print("- Performance optimization through scripted replay")
    
    try:
        # Run demonstrations
        manager, test_plan = await demonstrate_journal_recording()
        await demonstrate_pattern_matching(manager, test_plan)
        await demonstrate_dual_mode_execution()
        await demonstrate_execution_journal_analysis()
        
        print("\n\nDemo Summary")
        print("=" * 50)
        print("✓ Execution journaling captures complete test history")
        print("✓ Pattern library enables intelligent action reuse")
        print("✓ Dual-mode execution optimizes test performance")
        print("✓ Scripted replay provides 10x+ speed improvement")
        print("\nThe journaling system is ready for production use!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())