"""Enhanced HTML report generator."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Template

from haindy.core.types import TestState
from haindy.monitoring.enhanced_reporter_data import extract_template_data

logger = logging.getLogger(__name__)


def _load_template() -> Template:
    template_path = files("haindy.monitoring") / "templates" / "enhanced_report.html.j2"
    return Template(template_path.read_text(encoding="utf-8"))


class EnhancedReporter:
    """Enhanced HTML report generator with hierarchical structure."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._template = _load_template()

    def generate_report(
        self,
        test_state: TestState,
        output_dir: Path,
        action_storage: dict[str, Any] | None = None,
    ) -> tuple[Path, Path | None]:
        """
        Generate an enhanced HTML report.

        Args:
            test_state: The test state containing execution results
            output_dir: Directory to save the report
            action_storage: Optional action storage data

        Returns:
            Tuple of (report_path, actions_path)
        """
        if test_state.test_report is None:
            raise ValueError("Cannot generate enhanced report without a test report")

        template_data = extract_template_data(test_state, action_storage)
        html_content = self._template.render(**template_data)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        test_report = test_state.test_report
        filename = f"test_report_{test_report.test_plan_id}_{timestamp}.html"
        output_path = output_dir / filename
        output_path.write_text(html_content, encoding="utf-8")

        self.logger.info("Generated enhanced HTML report: %s", output_path)

        actions_path = None
        if action_storage and action_storage.get("test_cases"):
            actions_filename = f"{test_report.test_plan_id}_{timestamp}-actions.json"
            actions_path = output_dir / actions_filename
            with actions_path.open("w", encoding="utf-8") as handle:
                json.dump(action_storage, handle, indent=2, default=str)
            self.logger.info("Generated actions file: %s", actions_path)

        return output_path, actions_path
