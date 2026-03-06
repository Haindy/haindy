"""Summary helpers for TestRunner."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from src.core.types import BugSeverity, TestStatus, TestSummary


class TestRunnerSummary:
    """Owns summary calculation and summary-printing helpers."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def determine_overall_status(self) -> TestStatus:
        """Determine overall test execution status."""
        report = self._runner._test_report
        assert report is not None

        if not report.test_cases:
            return TestStatus.FAILED

        failed_cases = [
            test_case
            for test_case in report.test_cases
            if test_case.status == TestStatus.FAILED
        ]
        if failed_cases:
            return TestStatus.FAILED

        all_completed = all(
            test_case.status == TestStatus.PASSED for test_case in report.test_cases
        )
        if all_completed:
            return TestStatus.PASSED
        return TestStatus.SKIPPED

    def calculate_summary(self) -> TestSummary:
        """Calculate test execution summary statistics."""
        report = self._runner._test_report
        assert report is not None

        total_cases = (
            len(self._runner._current_test_plan.test_cases)
            if self._runner._current_test_plan is not None
            else len(report.test_cases)
        )
        executed_cases = len(report.test_cases)
        passed_cases = sum(
            1
            for test_case in report.test_cases
            if test_case.status in {TestStatus.PASSED, TestStatus.COMPLETED}
        )
        failed_cases = sum(
            1
            for test_case in report.test_cases
            if test_case.status == TestStatus.FAILED
        )
        skipped_cases = sum(
            1
            for test_case in report.test_cases
            if test_case.status in {TestStatus.SKIPPED, TestStatus.BLOCKED}
        )

        total_steps = sum(test_case.steps_total for test_case in report.test_cases)
        completed_steps = sum(
            test_case.steps_completed for test_case in report.test_cases
        )
        failed_steps = sum(test_case.steps_failed for test_case in report.test_cases)

        critical_bugs = sum(
            1 for bug in report.bugs if bug.severity == BugSeverity.CRITICAL
        )
        high_bugs = sum(1 for bug in report.bugs if bug.severity == BugSeverity.HIGH)
        medium_bugs = sum(
            1 for bug in report.bugs if bug.severity == BugSeverity.MEDIUM
        )
        low_bugs = sum(1 for bug in report.bugs if bug.severity == BugSeverity.LOW)

        if report.completed_at and report.started_at:
            execution_time = (report.completed_at - report.started_at).total_seconds()
        else:
            execution_time = 0.0

        success_rate = completed_steps / total_steps if total_steps > 0 else 0.0

        return TestSummary(
            total_test_cases=total_cases,
            completed_test_cases=executed_cases,
            passed_test_cases=passed_cases,
            failed_test_cases=failed_cases,
            skipped_test_cases=skipped_cases,
            total_steps=total_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            critical_bugs=critical_bugs,
            high_bugs=high_bugs,
            medium_bugs=medium_bugs,
            low_bugs=low_bugs,
            success_rate=success_rate,
            execution_time_seconds=execution_time,
        )

    @staticmethod
    def strip_plan_fingerprint_volatile_fields(payload: Any) -> Any:
        """Remove unstable fields from plan payloads before hashing."""
        if isinstance(payload, dict):
            stripped: dict[str, Any] = {}
            for key, value in payload.items():
                if key in {"plan_id", "created_at", "case_id", "step_id"}:
                    continue
                stripped[key] = (
                    TestRunnerSummary.strip_plan_fingerprint_volatile_fields(value)
                )
            return stripped
        if isinstance(payload, list):
            return [
                TestRunnerSummary.strip_plan_fingerprint_volatile_fields(item)
                for item in payload
            ]
        return payload

    def plan_fingerprint(self) -> str:
        """Build a stable fingerprint for replay cache keys."""
        if not self._runner._current_test_plan:
            return ""
        payload = self._runner._current_test_plan.model_dump(mode="json")
        stable_payload = self.strip_plan_fingerprint_volatile_fields(payload)
        serialized = json.dumps(
            stable_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    def print_summary(self) -> None:
        """Print test execution summary to console."""
        if not self._runner._test_report or not self._runner._test_report.summary:
            return

        summary = self._runner._test_report.summary

        print("\n" + "=" * 80)
        print(f"TEST EXECUTION SUMMARY: {self._runner._test_report.test_plan_name}")
        print("=" * 80)

        elapsed = int(summary.execution_time_seconds)
        elapsed_str = (
            f"{elapsed // 60}m{elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"
        )

        print(f"\nStatus: {self._runner._test_report.status.value.upper()}")
        print(
            f"Test Cases run: {summary.completed_test_cases}/{summary.total_test_cases}"
        )
        print(
            f"Test Cases passed: {summary.passed_test_cases}/{summary.total_test_cases}"
        )
        print(
            f"Test Cases failed: {summary.failed_test_cases}/{summary.total_test_cases}"
        )
        print(
            f"Test Cases skipped: {summary.skipped_test_cases}/{summary.total_test_cases}"
        )
        print(f"Steps: {summary.completed_steps}/{summary.total_steps} completed")
        print(f"Success Rate: {summary.success_rate * 100:.1f}%")
        print(f"Execution Time: {elapsed_str}")

        if self._runner._test_report.bugs:
            print(f"\nBugs Found: {len(self._runner._test_report.bugs)}")
            print(f"  Critical: {summary.critical_bugs}")
            print(f"  High: {summary.high_bugs}")
            print(f"  Medium: {summary.medium_bugs}")
            print(f"  Low: {summary.low_bugs}")

            critical = [
                bug
                for bug in self._runner._test_report.bugs
                if bug.severity == BugSeverity.CRITICAL
            ]
            if critical:
                print("\nCRITICAL BUGS:")
                for bug in critical:
                    print(f"  - {bug.description}")

        print("\n" + "=" * 80 + "\n")
