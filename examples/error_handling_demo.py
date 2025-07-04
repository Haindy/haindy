#!/usr/bin/env python3
"""
Demonstration of HAINDY's error handling and recovery capabilities.

This script showcases:
- Retry strategies with exponential backoff
- Hallucination detection in AI agent outputs
- Action validation before execution
- Error aggregation and reporting
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import ActionType, GridAction, ActionInstruction, GridCoordinate
from src.error_handling import (
    # Exceptions
    AgentError, BrowserError, ValidationError, TimeoutError,
    RetryableError, NonRetryableError, HallucinationError,
    
    # Recovery
    RecoveryManager, ExponentialBackoffStrategy, LinearBackoffStrategy,
    
    # Validation
    ActionValidator, ConfidenceScorer, HallucinationDetector,
    ValidationSeverity,
    
    # Aggregation
    ErrorAggregator, ErrorReport
)
from src.monitoring.logger import setup_logging

logger = setup_logging()


async def demonstrate_retry_strategies():
    """Demonstrate different retry strategies."""
    print("\n1. Retry Strategies Demo")
    print("=" * 50)
    
    recovery_manager = RecoveryManager()
    
    # Simulate a flaky operation
    attempt_count = 0
    async def flaky_browser_operation():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count < 3:
            print(f"  Attempt {attempt_count}: Operation failed (simulated)")
            raise BrowserError(
                "Element not found",
                url="https://example.com",
                selector="#submit-button",
                action="click"
            )
        else:
            print(f"  Attempt {attempt_count}: Operation succeeded!")
            return "Success"
    
    # Test exponential backoff
    print("\n  Exponential Backoff Strategy:")
    strategy = ExponentialBackoffStrategy(
        base_delay_ms=100,
        max_attempts=5,
        jitter=True
    )
    
    try:
        result = await recovery_manager.execute_with_recovery(
            flaky_browser_operation,
            "flaky_browser_op",
            retry_strategy=strategy
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed after all retries: {e}")
    
    # Test linear backoff
    print("\n  Linear Backoff Strategy:")
    attempt_count = 0  # Reset
    strategy = LinearBackoffStrategy(
        delay_increment_ms=200,
        max_attempts=4
    )
    
    try:
        result = await recovery_manager.execute_with_recovery(
            flaky_browser_operation,
            "flaky_browser_op_linear",
            retry_strategy=strategy
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Show statistics
    stats = recovery_manager.get_statistics()
    print("\n  Recovery Statistics:")
    for op_name, op_stats in stats.items():
        print(f"    {op_name}:")
        print(f"      - Successes: {op_stats['successes']}")
        print(f"      - Failures: {op_stats['failures']}")
        print(f"      - Total retries: {op_stats['retries']}")


async def demonstrate_hallucination_detection():
    """Demonstrate AI hallucination detection."""
    print("\n\n2. Hallucination Detection Demo")
    print("=" * 50)
    
    detector = HallucinationDetector()
    
    # Test cases
    test_cases = [
        {
            "agent_output": "I can see the 'Purchase Now' button at the top of the page",
            "screenshot_elements": {"Add to Cart", "View Details", "Back to Home"},
            "description": "Agent claims to see non-existent button"
        },
        {
            "agent_output": "Clicking on the Add to Cart button",
            "screenshot_elements": {"Add to Cart", "Remove", "Quantity"},
            "description": "Agent correctly identifies existing element"
        },
        {
            "agent_output": "The element is located at coordinates: (-50, 200)",
            "viewport_size": (1920, 1080),
            "description": "Agent provides negative coordinates"
        },
        {
            "agent_output": "Found the button at coordinates: (1800, 1000)",
            "viewport_size": (1920, 1080),
            "description": "Agent provides valid coordinates"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test_case['description']}")
        print(f"  Agent output: \"{test_case['agent_output']}\"")
        
        error = detector.detect_hallucinations(
            test_case["agent_output"],
            "test_agent",
            screenshot_elements=test_case.get("screenshot_elements"),
            viewport_size=test_case.get("viewport_size")
        )
        
        if error:
            print(f"  ❌ Hallucination detected!")
            print(f"     Type: {error.hallucination_type}")
            print(f"     Confidence: {error.confidence_score:.2f}")
            print(f"     Evidence: {error.evidence}")
        else:
            print(f"  ✓ No hallucination detected")


async def demonstrate_action_validation():
    """Demonstrate action validation before execution."""
    print("\n\n3. Action Validation Demo")
    print("=" * 50)
    
    validator = ActionValidator()
    scorer = ConfidenceScorer()
    
    # Test actions using proper GridAction structure
    actions = [
        {
            "action": GridAction(
                instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click submit button",
                    target="#valid-button",
                    expected_outcome="Button clicked"
                ),
                coordinate=GridCoordinate(
                    cell="H15",
                    offset_x=0.7,
                    offset_y=0.4,
                    confidence=0.92,
                    refined=True
                )
            ),
            "context": {
                "viewport_size": (1920, 1080),
                "page_loaded": True,
                "element_exists": True
            },
            "description": "Valid action with good confidence"
        },
        {
            "action": GridAction(
                instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click phantom button",
                    expected_outcome="Button clicked"
                ),
                coordinate=GridCoordinate(
                    cell="ZZ99",  # Invalid grid position
                    confidence=0.15,  # Low confidence
                    refined=False
                )
            ),
            "context": {
                "viewport_size": (1920, 1080),
                "page_loaded": True
            },
            "description": "Low confidence coordinates"
        },
        {
            "action": GridAction(
                instruction=ActionInstruction(
                    action_type=ActionType.TYPE,
                    description="Type in field",
                    target="   ",  # Empty target
                    expected_outcome="Text entered"
                ),
                coordinate=GridCoordinate(
                    cell="M20",
                    confidence=0.5
                )
            ),
            "context": {
                "page_loaded": True,
                "element_exists": False
            },
            "description": "Missing selector and element"
        }
    ]
    
    for i, test in enumerate(actions, 1):
        print(f"\n  Test {i}: {test['description']}")
        
        # Calculate confidence
        confidence = scorer.calculate_action_confidence(
            test["action"],
            screenshot_analysis={"confidence": 0.8}
        )
        confidence_level = scorer.get_confidence_level(confidence)
        
        print(f"  Confidence: {confidence:.2f} ({confidence_level})")
        
        # Validate action
        is_valid, results = await validator.validate_action(
            test["action"],
            test["context"]
        )
        
        print(f"  Validation: {'✓ PASSED' if is_valid else '❌ FAILED'}")
        
        # Show validation details
        for result in results:
            if not result.is_valid:
                print(f"    - {result.rule_name}: {result.message}")


async def demonstrate_error_aggregation():
    """Demonstrate error aggregation and reporting."""
    print("\n\n4. Error Aggregation Demo")
    print("=" * 50)
    
    aggregator = ErrorAggregator("demo_test_123", "Error Handling Demo")
    
    # Simulate test execution with various errors
    print("\n  Simulating test execution with errors...")
    
    # Agent errors
    for i in range(4):
        error = AgentError(
            f"Agent processing failed #{i+1}",
            agent_name=f"agent_{i % 2}",
            agent_type="TestRunnerAgent"
        )
        aggregator.add_error(
            error,
            agent_name=f"agent_{i % 2}",
            operation="process_instruction",
            recovered=(i < 2)  # First 2 recovered
        )
    
    # Browser errors
    for i in range(3):
        error = BrowserError(
            f"Browser automation failed #{i+1}",
            url="https://test.com",
            action="click"
        )
        aggregator.add_error(
            error,
            operation="click_element",
            recovered=True  # All recovered
        )
    
    # Timeout errors
    for i in range(2):
        error = TimeoutError(
            "Operation timed out",
            operation="page_load",
            timeout_ms=5000
        )
        aggregator.add_error(
            error,
            operation="load_page",
            recovered=False  # None recovered
        )
    
    # Validation error
    aggregator.add_error(
        ValidationError(
            "Invalid action parameters",
            validation_type="action_validation",
            failed_rules=["coordinate_bounds", "element_selector"]
        ),
        operation="validate_action"
    )
    
    # Generate report
    report = aggregator.generate_report()
    
    print(f"\n  Test Summary:")
    print(f"    - Test ID: {report.test_id}")
    print(f"    - Duration: {(report.end_time - report.start_time).total_seconds():.2f}s")
    print(f"    - Total errors: {report.total_errors}")
    
    print(f"\n  Errors by Category:")
    for category, count in report.errors_by_category.items():
        print(f"    - {category.name}: {count}")
    
    print(f"\n  Critical Errors:")
    if report.critical_errors:
        for error in report.critical_errors:
            print(f"    - {error['error_type']}: {error['count']} occurrences")
            print(f"      Recovery rate: {error['recovery_rate']:.1%}")
    else:
        print("    None (no errors exceeded critical threshold)")
    
    print(f"\n  Recovery Summary:")
    summary = report.recovery_summary
    print(f"    - Total attempts: {summary['total_recovery_attempts']}")
    print(f"    - Successful recoveries: {summary['total_recovery_successes']}")
    print(f"    - Overall recovery rate: {summary['overall_recovery_rate']:.1%}")
    if summary['best_recovery']:
        print(f"    - Best recovery: {summary['best_recovery']}")
    if summary['worst_recovery']:
        print(f"    - Worst recovery: {summary['worst_recovery']}")
    
    print(f"\n  Recommendations:")
    if report.recommendations:
        for rec in report.recommendations:
            print(f"    - {rec}")
    else:
        print("    No specific recommendations")
    
    # Save report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report.save_to_file(report_file)
    print(f"\n  Report saved to: {report_file}")


async def demonstrate_fallback_mechanisms():
    """Demonstrate fallback mechanisms."""
    print("\n\n5. Fallback Mechanisms Demo")
    print("=" * 50)
    
    recovery_manager = RecoveryManager()
    
    # Primary operation that always fails
    async def primary_operation():
        print("  Primary: Attempting browser automation...")
        raise BrowserError(
            "Element not clickable",
            selector="#dynamic-button",
            max_retries=1  # Only retry once
        )
    
    # Fallback using grid coordinates
    async def fallback_operation():
        print("  Fallback: Using grid-based interaction...")
        return "Clicked via grid coordinates"
    
    # Execute with fallback
    print("\n  Executing operation with fallback strategy:")
    
    strategy = LinearBackoffStrategy(
        delay_increment_ms=100,
        max_attempts=2
    )
    
    try:
        result = await recovery_manager.execute_with_recovery(
            primary_operation,
            "click_with_fallback",
            retry_strategy=strategy,
            fallback=fallback_operation
        )
        print(f"  Success: {result}")
    except Exception as e:
        print(f"  Failed: {e}")


async def main():
    """Run all demonstrations."""
    print("\nHAINDY Error Handling & Recovery Demo")
    print("=" * 70)
    print("\nThis demo showcases HAINDY's robust error handling capabilities:")
    print("- Intelligent retry strategies with backoff")
    print("- AI hallucination detection")
    print("- Pre-execution action validation")
    print("- Comprehensive error aggregation and reporting")
    print("- Fallback mechanisms for resilient test execution")
    
    try:
        await demonstrate_retry_strategies()
        await demonstrate_hallucination_detection()
        await demonstrate_action_validation()
        await demonstrate_error_aggregation()
        await demonstrate_fallback_mechanisms()
        
        print("\n\nDemo Summary")
        print("=" * 50)
        print("✓ Retry strategies ensure resilient test execution")
        print("✓ Hallucination detection prevents false positives")
        print("✓ Action validation catches errors before execution")
        print("✓ Error aggregation provides actionable insights")
        print("✓ Fallback mechanisms maximize test success rates")
        print("\nThe error handling system is ready for production use!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())