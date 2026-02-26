import logging
from typing_extensions import override
from pathlib import Path

from rag_eval.components.report_generator import ReportGenerator
from rag_eval.schemas.eval_schemas import EvalReport

logger = logging.getLogger(__name__)

class EvalReportGenerator(ReportGenerator[EvalReport]):
    """
    Generates timestamped JSON and Markdown reports for a single RAG pipeline evaluation run.

    Inherits shared file I/O infrastructure from ReportGenerator. Implements
    _write_markdown to produce an EvalReport-specific Markdown report containing
    a summary table of aggregate metrics (context precision and recall) and
    per-question detail including individual scores and retrieved contexts.
    """
    @override
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