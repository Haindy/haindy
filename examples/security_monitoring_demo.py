#!/usr/bin/env python3
"""
Demonstration of HAINDY's security and monitoring capabilities.

This script showcases:
- Rate limiting for API and browser actions
- Sensitive data sanitization in logs
- Analytics and metrics collection
- Test execution reporting
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security import (
    RateLimiter, RateLimitConfig, RateLimitExceeded,
    DataSanitizer, SensitiveDataPattern, RedactionMethod
)
from src.monitoring import (
    setup_logging, get_logger,
    start_test, end_test, record_step, record_api_call,
    TestOutcome, get_analytics,
    ReportGenerator
)
from src.error_handling import ErrorAggregator


async def demonstrate_rate_limiting():
    """Demonstrate rate limiting functionality."""
    print("\n1. Rate Limiting Demo")
    print("=" * 50)
    
    # Configure rate limiter
    config = RateLimitConfig(
        api_calls_per_minute=30,  # 0.5 per second
        api_burst_size=3,
        browser_actions_per_minute=60,  # 1 per second
        browser_actions_burst=5
    )
    
    limiter = RateLimiter(config)
    
    print("\n  API Rate Limiting (30/min, burst of 3):")
    
    # Demonstrate burst capability
    for i in range(5):
        try:
            allowed = await limiter.check_api_call()
            if allowed:
                print(f"    API call {i+1}: ✓ Allowed")
            else:
                print(f"    API call {i+1}: ✗ Blocked")
        except RateLimitExceeded as e:
            print(f"    API call {i+1}: ✗ Rate limit exceeded (retry after {e.retry_after:.1f}s)")
    
    print("\n  Waiting for token refill...")
    await asyncio.sleep(2)  # Wait for tokens to refill
    
    # Should allow more calls
    allowed = await limiter.check_api_call()
    print(f"    API call after wait: {'✓ Allowed' if allowed else '✗ Blocked'}")
    
    # Show statistics
    stats = limiter.get_statistics()
    api_stats = stats.get("api", {})
    print(f"\n  API Rate Limit Statistics:")
    print(f"    - Allowed: {api_stats.get('allowed', 0)}")
    print(f"    - Rejected: {api_stats.get('rejected', 0)}")
    print(f"    - Rejection rate: {api_stats.get('rejection_rate', 0):.1%}")


def demonstrate_data_sanitization():
    """Demonstrate sensitive data protection."""
    print("\n\n2. Data Sanitization Demo")
    print("=" * 50)
    
    # Create logger with sanitization
    logger = setup_logging(log_format="text", sanitize_logs=True)
    demo_logger = get_logger("demo.sanitizer")
    
    print("\n  Testing various sensitive data patterns:")
    
    # Test data with sensitive information
    test_cases = [
        {
            "message": "User login with email user@example.com",
            "description": "Email address"
        },
        {
            "message": "API call with key: sk_test_123456789abcdef",
            "description": "API key"
        },
        {
            "message": "Credit card payment: 4111-1111-1111-1111",
            "description": "Credit card number"
        },
        {
            "message": "Password reset for password=SuperSecret123!",
            "description": "Password field"
        },
        {
            "message": "SSN verification: 123-45-6789",
            "description": "Social Security Number"
        }
    ]
    
    sanitizer = DataSanitizer()
    
    for test in test_cases:
        print(f"\n  {test['description']}:")
        print(f"    Original: {test['message']}")
        sanitized = sanitizer.sanitize_string(test['message'])
        print(f"    Sanitized: {sanitized}")
        
        # Also log it (will be sanitized by logger)
        demo_logger.info(test['message'])
    
    # Test dictionary sanitization
    print("\n  Dictionary sanitization:")
    sensitive_data = {
        "user": {
            "email": "john.doe@company.com",
            "api_key": "sk_live_super_secret_key_123",
            "profile": {
                "phone": "555-123-4567",
                "address": "123 Main St"  # Not sensitive
            }
        },
        "payment": {
            "card_last_four": "1234",
            "card_full": "4532-1234-5678-9012"
        }
    }
    
    print("    Original data:")
    print(f"      {sensitive_data}")
    
    sanitized_data = sanitizer.sanitize_dict(sensitive_data)
    print("\n    Sanitized data:")
    print(f"      {sanitized_data}")


async def demonstrate_analytics_and_reporting():
    """Demonstrate analytics collection and reporting."""
    print("\n\n3. Analytics and Reporting Demo")
    print("=" * 50)
    
    # Get analytics collector
    analytics = get_analytics()
    analytics.reset()  # Clear any previous data
    
    # Simulate test execution
    print("\n  Simulating test execution with metrics...")
    
    test_id = uuid4()
    test_name = "demo_checkout_flow"
    
    # Start test
    await start_test(test_id, test_name)
    print(f"    Started test: {test_name}")
    
    # Simulate steps with metrics
    steps = [
        ("Navigate to product page", True, 500, "navigate"),
        ("Search for product", True, 300, "search"),
        ("Click add to cart", True, 150, "click"),
        ("View cart", True, 200, "navigate"),
        ("Enter shipping info", False, 1200, "form"),  # Failed
        ("Retry shipping info", True, 800, "form"),
        ("Enter payment", True, 600, "form"),
        ("Submit order", True, 400, "click")
    ]
    
    for step_name, success, duration_ms, action_type in steps:
        # Record step
        await record_step(test_id, step_name, success, duration_ms)
        
        # Record browser action
        await analytics.record_browser_action(test_id, action_type, duration_ms, success)
        
        # Simulate API calls for some steps
        if action_type in ["search", "form"]:
            await record_api_call(test_id, "backend_api", duration_ms // 2)
        
        print(f"    Step: {step_name} - {'✓' if success else '✗'} ({duration_ms}ms)")
        
        # Small delay to simulate real execution
        await asyncio.sleep(0.1)
    
    # End test
    await end_test(test_id, TestOutcome.PASSED)
    print(f"    Test completed: PASSED")
    
    # Get summaries
    print("\n  Test Summary:")
    test_summary = analytics.get_test_summary()
    print(f"    - Total tests: {test_summary['total_tests']}")
    print(f"    - Success rate: {test_summary['success_rate']:.1%}")
    print(f"    - Average duration: {test_summary['avg_duration']:.2f}s")
    
    print("\n  Performance Summary:")
    perf_summary = analytics.get_performance_summary()
    print(f"    - API calls/min: {perf_summary['api_calls']['rate_per_minute']:.1f}")
    print(f"    - Browser actions/min: {perf_summary['browser_actions']['rate_per_minute']:.1f}")
    print(f"    - Step success rate: {perf_summary['steps']['success_rate']:.1%}")
    
    # Generate report
    print("\n  Generating test report...")
    
    # Create error report (minimal for demo)
    error_aggregator = ErrorAggregator(str(test_id), test_name)
    error_report = error_aggregator.generate_report()
    
    # Generate test report
    report_generator = ReportGenerator(analytics, output_dir=Path("reports"))
    test_report = report_generator.generate_test_report(test_id, error_report)
    
    if test_report:
        # Save reports
        saved_files = test_report.save(
            Path("reports"),
            formats=["json", "html", "markdown"]
        )
        
        print("    Reports saved:")
        for format_type, filepath in saved_files.items():
            print(f"      - {format_type}: {filepath}")


async def demonstrate_integrated_monitoring():
    """Demonstrate integrated monitoring features."""
    print("\n\n4. Integrated Monitoring Demo")
    print("=" * 50)
    
    # Set up monitoring with all features
    print("\n  Setting up integrated monitoring...")
    
    # Configure components
    rate_config = RateLimitConfig(
        api_calls_per_minute=60,
        browser_actions_per_minute=120
    )
    
    limiter = RateLimiter(rate_config)
    sanitizer = DataSanitizer()
    analytics = get_analytics()
    
    # Add custom sanitization pattern
    sanitizer.add_pattern(SensitiveDataPattern(
        name="internal_id",
        pattern=re.compile(r'ID-\d{6}'),
        redaction_method=RedactionMethod.HASH,
        description="Internal ID pattern"
    ))
    
    logger = get_logger("demo.integrated")
    
    print("    ✓ Rate limiter configured")
    print("    ✓ Data sanitizer configured")
    print("    ✓ Analytics collector ready")
    print("    ✓ Logging with sanitization enabled")
    
    # Simulate monitored operation
    print("\n  Simulating monitored test operation...")
    
    test_id = uuid4()
    await start_test(test_id, "monitored_test")
    
    # Simulate rate-limited API calls
    print("\n    Making rate-limited API calls:")
    for i in range(3):
        try:
            await limiter.check_api_call()
            await record_api_call(test_id, "protected_api", 100)
            logger.info(f"API call {i+1} successful with ID-{123456+i}")
        except RateLimitExceeded:
            logger.warning(f"API call {i+1} rate limited")
    
    # Log with sensitive data (will be sanitized)
    logger.info(
        "Test user user@test.com authenticated with token Bearer abc123def456",
        extra={"internal_id": "ID-789012", "test_id": str(test_id)}
    )
    
    await end_test(test_id, TestOutcome.PASSED)
    
    # Show final statistics
    print("\n  Final Statistics:")
    
    rate_stats = limiter.get_statistics()
    print(f"    Rate limiting:")
    for limit_type, stats in rate_stats.items():
        if stats.get('allowed', 0) > 0:
            print(f"      - {limit_type}: {stats['allowed']} allowed, {stats['rejected']} rejected")
    
    sanitizer_stats = sanitizer.get_statistics()
    print(f"\n    Sanitization:")
    print(f"      - Active patterns: {sanitizer_stats['patterns']['enabled']}")
    print(f"      - Active rules: {sanitizer_stats['rules']['enabled']}")
    
    test_stats = analytics.get_test_summary()
    print(f"\n    Test execution:")
    print(f"      - Tests run: {test_stats['total_tests']}")
    print(f"      - Success rate: {test_stats['success_rate']:.1%}")


async def main():
    """Run all demonstrations."""
    print("\nHAINDY Security & Monitoring Demo")
    print("=" * 70)
    print("\nThis demo showcases HAINDY's security and monitoring capabilities:")
    print("- Rate limiting for controlled resource usage")
    print("- Sensitive data sanitization in logs and reports")
    print("- Comprehensive metrics and analytics collection")
    print("- Automated test execution reporting")
    
    try:
        # Run demonstrations
        await demonstrate_rate_limiting()
        demonstrate_data_sanitization()
        await demonstrate_analytics_and_reporting()
        await demonstrate_integrated_monitoring()
        
        print("\n\nDemo Summary")
        print("=" * 50)
        print("✓ Rate limiting prevents API abuse and ensures stability")
        print("✓ Data sanitization protects sensitive information")
        print("✓ Analytics provide insights into test performance")
        print("✓ Integrated monitoring ensures secure, observable execution")
        print("\nThe security and monitoring system is ready for production use!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Need to import re for custom pattern
    import re
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Run demo
    asyncio.run(main())