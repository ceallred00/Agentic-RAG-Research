from datetime import datetime
from pathlib import Path
from typing import Union, Tuple
import logging

from rag_eval.schemas.eval_schemas import EvalReport
from constants import RAG_EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: Union[str, Path] = RAG_EVAL_RESULTS_DIR):
        self.output_directory = Path(output_dir)
    def generate_report(self, report: EvalReport)-> Tuple[Path, Path]:
        """
        Generates a JSON report and human-readable MD report from the RAGAS Evaluation Report.
        
        Raises: 
            FileNotFoundError - If the output directory does not exist.
        """
        if not self.output_directory.exists():
            error_msg = "Report output directory not found."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        json_path, md_path, timestamp = self._generate_filepaths()

        self._write_JSON(report = report, json_output_path = json_path)
        self._write_markdown(report = report, md_output_path = md_path, timestamp = timestamp)
        return json_path, md_path
    def _generate_filepaths(self) -> Tuple[Path, Path, str]:
        """Generates a custom filepath using the current timestamp."""
        logger.info("Generating filepaths...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{timestamp}"
        json_path = self.output_directory / f"{filename}.json"
        md_path = self.output_directory / f"{filename}.md"
        logger.info(f"Two filepaths generated: {json_path, md_path}")
        return json_path, md_path, timestamp
    def _write_JSON(self, report: EvalReport, json_output_path: Path):
        """Converts EvalReport Pydantic Model to JSON."""
        # Gives a formatted JSON string
        logger.info("Generating JSON from RAGAS metrics")
        try:
            json_report = report.model_dump_json(indent=2)
            json_output_path.write_text(data = json_report, encoding = 'utf-8')
        except OSError as e:
            error_msg = f"Failed to write report to {json_output_path}: {e}"
            logger.error(error_msg)
            raise OSError(error_msg) from e

    def _write_markdown(self, report: EvalReport, md_output_path: Path, timestamp: str):
        """Converts EvalReport Pydantic Model to Human Readable Markdown"""
        logger.info("Generating MD from RAGAS metrics.")
        md = "# RAG Evaluation Report\n"
        md += f"## {report.description}\n\n"
        md += f"**Dataset:** {report.dataset_name}\n"
        md += f"**Timestamp:** {timestamp}\n"
        md += f"**Total Questions:** {report.total_questions_evaluated}\n\n"
        md += "## Summary\n"
        md += "| Metric | Score |\n"
        md += "|--------|-------|\n"
        md += f"| Avg Context Precision | {report.average_context_precision:.2f} |\n"
        md += f"| Avg Context Recall | {report.average_context_recall:.2f} |\n\n"

        results_md = "## Per-Question Results\n"

        # List of evaluation results for each question
        for index, result in enumerate(report.per_question_results, 1):
            results_md += f"### Q{index}: {result.question}\n"
            results_md += f"- Context Precision: {result.context_precision:.2f}\n"
            results_md += f"- Context Recall: {result.context_recall:.2f}\n"
            results_md += "- Retrieved Contexts:\n"
            for i, context in enumerate(result.contexts, 1):
                results_md += f"    {i}. {context}\n"
        
        md += results_md

        try:
            md_output_path.write_text(md, encoding = 'utf-8')
        
        except OSError as e:
            error_msg = f"Failed to write report to {md_output_path}: {e}"
            logger.error(error_msg)
            raise OSError(error_msg) from e