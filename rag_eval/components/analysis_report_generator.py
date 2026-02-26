import csv
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import logging
from typing_extensions import override
from pydantic import ValidationError

from constants import RAG_EVAL_RESULTS_DIR, RAG_EVAL_ANALYSIS_DIR, RAG_EVAL_TRACKER_PATH
from rag_eval.components.report_generator import ReportGenerator
from rag_eval.schemas.eval_schemas import EvalReport, QuestionEvalResult
from rag_eval.schemas.analysis_schemas import EvalReportSummary, AggregatedPoorResult
from rag_eval.utils.compute_aggregate_metrics import compute_average, compute_standard_deviation


logger = logging.getLogger(__name__)


class AnalysisReportGenerator(ReportGenerator[EvalReportSummary]):
    """
    Generates cross-run analysis reports by aggregating results across multiple
    RAG evaluation pipeline runs.

    Loads EvalReport JSON files, identifies questions that fell below a performance
    threshold, and produces a JSON and Markdown summary report containing cross-run
    aggregate metrics and per-question diagnostic detail.

    Inherits shared file I/O infrastructure from ReportGenerator. Implements
    _write_markdown to produce an EvalReportSummary-specific Markdown report.
    """
    def __init__(self,
                 output_dir=RAG_EVAL_ANALYSIS_DIR,
                 results_dir=RAG_EVAL_RESULTS_DIR,
                 tracker_path=RAG_EVAL_TRACKER_PATH):
        super().__init__(output_dir=output_dir, prefix="analysis")
        self.results_directory = Path(results_dir)
        self.tracker_path = Path(tracker_path)

    @staticmethod
    def _load_eval_report(file_path: Path) -> EvalReport:
        """
        Reads a JSON file and deserializes it into an EvalReport.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return EvalReport.model_validate_json(f.read())

    @staticmethod
    def _extract_poor_results(eval_report: EvalReport,
                              threshold: float = 0.5
                              ) -> Tuple[List[QuestionEvalResult], List[QuestionEvalResult]]:
        """
        Extracts questions with scores below the given threshold.

        Returns a tuple of (poor_recall_results, poor_precision_results),
        each a list of QuestionEvalResult objects below the threshold.
        """
        poor_recall_scores: List[QuestionEvalResult] = []
        poor_precision_scores: List[QuestionEvalResult] = []

        per_question_results = eval_report.per_question_results
        if not per_question_results:
            raise ValueError("EvalReport contains no per_question_results.")

        for result in per_question_results:
            if result.context_precision < threshold:
                poor_precision_scores.append(result)
            if result.context_recall < threshold:
                poor_recall_scores.append(result)

        logger.info(f"Extracted {len(poor_recall_scores)} questions with poor recall scores.")
        logger.info(f"Extracted {len(poor_precision_scores)} questions with poor precision scores.")

        return poor_recall_scores, poor_precision_scores

    @staticmethod
    def _process_poor_results(
        results: List[QuestionEvalResult],
        accumulator: Dict[str, Dict[str, List]],
        score_fn: Callable[[QuestionEvalResult], float]
    ) -> None:
        """
        Accumulates poor-performing question results into a dict keyed by question string.

        For each result, extracts the score using score_fn and appends the score and
        retrieved contexts to the accumulator. If the question is seen for the first time,
        a new entry is created. Subsequent runs append to the existing entry, allowing
        cross-run aggregation grouped by question.

        Args:
            results: List of QuestionEvalResult objects below the threshold for a single run.
            accumulator: Dict keyed by question string, accumulating scores and contexts across runs.
            score_fn: Callable that extracts the relevant metric score from a QuestionEvalResult
                      (e.g., lambda r: r.context_recall).
        """
        for result in results:
            question = result.question
            score = score_fn(result)
            contexts = result.contexts
            if question not in accumulator:
                accumulator[question] = {"scores": [score], "contexts": [contexts]}
            else:
                accumulator[question]["scores"].append(score)
                accumulator[question]["contexts"].append(contexts)

    @staticmethod
    def _convert_to_aggregated_poor_result(metric_dict: Dict[str, Dict[str, List]]) -> List[AggregatedPoorResult]:
        """
        Converts the cross-run accumulator dict into a list of AggregatedPoorResult objects.

        Each entry in the dict corresponds to a question that fell below the threshold in at
        least one run. The resulting AggregatedPoorResult captures the per-run scores, their
        average, the number of failing runs, and the contexts retrieved in each failing run.

        Args:
            metric_dict: Dict keyed by question string, where each value contains
                         "scores" (List[float]) and "contexts" (List[List[str]]).
        """
        results_list: List[AggregatedPoorResult] = []
        for question, val in metric_dict.items():
            scores = val["scores"]
            results_list.append(AggregatedPoorResult(
                question=question,
                scores=scores,
                avg_score=compute_average(scores),
                runs_below_threshold=len(scores),
                contexts_per_run=val["contexts"]
            ))
        return results_list

    @override
    def _write_markdown(self,
                        report: EvalReportSummary,
                        md_output_path: Path,
                        timestamp: str):
        """Converts EvalReportSummary Pydantic model to a human-readable cross-run analysis Markdown report."""
        logger.info("Generating MD from EvalReportSummary.")
        md = "# RAG Evaluation Analysis Report\n"
        md += f"## {report.description}\n\n"
        md += f"**Dataset:** {report.dataset_name}\n"
        md += f"**Timestamp:** {timestamp}\n"
        md += f"**Reports Analyzed:** {len(report.file_names)}\n\n"
        md += "**Report Files:**\n"
        for filename in report.file_names:
            md += f"- {filename}\n"
        md += "\n"

        md += "## Cross-Run Metrics Summary\n"
        md += "| Metric | Mean | Std Dev |\n"
        md += "|--------|------|----------|\n"
        md += f"| Context Recall | {report.cross_run_average_context_recall:.2f} | {report.cross_run_std_context_recall:.2f} |\n"
        md += f"| Context Precision | {report.cross_run_average_context_precision:.2f} | {report.cross_run_std_context_precision:.2f} |\n\n"

        md += "## Per-Run Metrics\n"
        md += "| Run | File | Avg Recall | Avg Precision |\n"
        md += "|-----|------|------------|---------------|\n"
        for run_idx, (filename, recall, precision) in enumerate(
            zip(report.file_names, report.per_run_context_recall, report.per_run_context_precision), 1
        ):
            md += f"| {run_idx} | {filename} | {recall:.2f} | {precision:.2f} |\n"
        md += "\n"

        def format_poor_results(results: List[AggregatedPoorResult], preserve_order: bool) -> str:
            section = ""
            for index, result in enumerate(results, 1):
                section += f"### Q{index}: {result.question}\n"
                section += f"- **Avg Score:** {result.avg_score:.2f} | **Runs Below Threshold:** {result.runs_below_threshold}\n"
                if preserve_order:
                    # Precision: ranked order per run is diagnostically significant.
                    # Render each run separately to show how contexts were ranked.
                    for run_idx, contexts in enumerate(result.contexts_per_run, 1):
                        section += f"- **Run {run_idx} Contexts (ranked):**\n\n"
                        for i, context in enumerate(contexts, 1):
                            section += f"{i}. {context}\n\n"
                else:
                    # Recall: order is irrelevant. Deduplicate contexts across runs
                    # and annotate each with the run numbers that retrieved it.
                    context_to_runs: Dict[str, List[int]] = {}
                    for run_idx, contexts in enumerate(result.contexts_per_run, 1):
                        for context in contexts:
                            if context not in context_to_runs:
                                context_to_runs[context] = []
                            context_to_runs[context].append(run_idx)
                    section += "- **Retrieved Contexts:**\n\n"
                    for i, (context, runs) in enumerate(context_to_runs.items(), 1):
                        run_label = "All runs" if len(runs) == result.runs_below_threshold else f"Run{'s' if len(runs) > 1 else ''}: {', '.join(str(r) for r in runs)}"
                        section += f"This context appeared in: {run_label}\n"
                        section += f"{i}. {context}\n"
                section += "\n"
            return section

        md += "## Questions With Poor Recall\n"
        if report.poor_recall_results:
            md += format_poor_results(report.poor_recall_results, preserve_order=False)
        else:
            md += "_No questions below the recall threshold._\n"
        md += "\n"

        md += "## Questions With Poor Precision\n"
        if report.poor_precision_results:
            md += format_poor_results(report.poor_precision_results, preserve_order=True)
        else:
            md += "_No questions below the precision threshold._\n"

        try:
            md_output_path.write_text(md, encoding="utf-8")
        except OSError as e:
            error_msg = f"Failed to write report to {md_output_path}: {e}"
            logger.error(error_msg)
            raise OSError(error_msg) from e

    def _append_to_tracker(self, summary: EvalReportSummary, timestamp: str) -> None:
        """
        Appends a row of aggregate metrics to the cross-run metrics tracker CSV.

        If the tracker file does not exist, it is created with a header row first.
        Each call appends one row corresponding to the current analysis run.

        Args:
            summary: The EvalReportSummary containing the aggregate metrics to log.
            timestamp: Formatted timestamp string (YYYYMMDD_HHMMSS) identifying the run.

        Raises:
            OSError: If the tracker file cannot be written.
        """
        write_header = not self.tracker_path.exists()
        row = {
            "timestamp": timestamp,
            "dataset_name": summary.dataset_name,
            "description": summary.description,
            "reports_analyzed": len(summary.file_names),
            "avg_recall": round(summary.cross_run_average_context_recall, 4),
            "avg_precision": round(summary.cross_run_average_context_precision, 4),
            "std_recall": round(summary.cross_run_std_context_recall, 4),
            "std_precision": round(summary.cross_run_std_context_precision, 4),
        }
        try:
            with open(self.tracker_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            logger.info(f"Appended metrics to tracker at {self.tracker_path}.")
        except OSError as e:
            error_msg = f"Failed to write to tracker at {self.tracker_path}: {e}"
            logger.error(error_msg)
            raise OSError(error_msg) from e

    def analyze_and_report(self,
                           report_paths: List[Path],
                           dataset_name: str,
                           dataset_description: str,
                           threshold: float = 0.5
                           ) -> Tuple[Path, Path]:
        """
        Loads and aggregates evaluation reports across multiple pipeline runs, then generates
        a JSON and Markdown summary report.

        For each report path, loads the EvalReport, extracts per-question results below the
        threshold, and accumulates them across runs. After all reports are processed, computes
        cross-run average and standard deviation for context recall and precision, and
        identifies questions that consistently underperformed.

        Raises:
            FileNotFoundError: If a report file does not exist (fail-fast).
            ValueError: If no reports were successfully processed.

        Args:
            report_paths: List of paths to JSON evaluation report files.
            dataset_name: Name of the dataset being analyzed (e.g., "KB Testing Dataset").
            dataset_description: Brief description of the evaluation configuration.
            threshold: Score threshold below which a result is considered poor. Defaults to 0.5.

        Returns:
            Tuple of (json_output_path, md_output_path) for the generated summary report.
        """
        cross_run_recall_scores: List[float] = []
        cross_run_precision_scores: List[float] = []
        filenames: List[str] = []

        poor_recall_dict: Dict[str, Dict[str, List]] = {}
        poor_precision_dict: Dict[str, Dict[str, List]] = {}

        for report_path in report_paths:
            # Load EvalReport representing a single pipeline iteration.
            try:
                report_data = self._load_eval_report(report_path)
            except FileNotFoundError as e:
                error_msg = f"{report_path} not found: {e}."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            except ValidationError as e:
                logger.warning(f"Skipping {report_path}: malformed report. {e}")
                continue

            # Iterate over each pipeline run, extract poor results
            try:
                poor_recall_results, poor_precision_results = self._extract_poor_results(
                    eval_report=report_data,
                    threshold=threshold)
            except ValueError as e:
                logger.warning(f"{report_path} contains no per_question_results.")
                continue

            # Modifies metric dictionaries in place
            self._process_poor_results(poor_recall_results, poor_recall_dict, lambda r: r.context_recall)
            self._process_poor_results(poor_precision_results, poor_precision_dict, lambda r: r.context_precision)

            # Append filename once all extractions succeed.
            filenames.append(report_path.stem)
            cross_run_recall_scores.append(report_data.average_context_recall)
            cross_run_precision_scores.append(report_data.average_context_precision)

        if not filenames:
            raise ValueError("No reports were successfully processed. Cannot generate summary.")

        summary = EvalReportSummary(
            file_names=filenames,
            cross_run_average_context_recall=compute_average(cross_run_recall_scores),
            cross_run_average_context_precision=compute_average(cross_run_precision_scores),
            cross_run_std_context_precision=compute_standard_deviation(cross_run_precision_scores),
            cross_run_std_context_recall=compute_standard_deviation(cross_run_recall_scores),
            per_run_context_recall=cross_run_recall_scores,
            per_run_context_precision=cross_run_precision_scores,
            dataset_name=dataset_name,
            description=dataset_description,
            poor_recall_results=self._convert_to_aggregated_poor_result(poor_recall_dict),
            poor_precision_results=self._convert_to_aggregated_poor_result(poor_precision_dict)
        )

        json_path, md_path = self.generate_report(summary)
        timestamp = json_path.stem.split("_", 1)[1]
        self._append_to_tracker(summary, timestamp)
        return json_path, md_path


if __name__ == "__main__":  # pragma: no cover
    generator = AnalysisReportGenerator(output_dir=RAG_EVAL_ANALYSIS_DIR, results_dir=RAG_EVAL_RESULTS_DIR)
    report_paths = [
        Path("/Users/jacoblopez/Documents/Cheyanne/MS in Data Science/AI Agents/rag_eval/results/eval_20260225_142216.json"),
        Path("/Users/jacoblopez/Documents/Cheyanne/MS in Data Science/AI Agents/rag_eval/results/eval_20260225_144537.json"),
        Path("/Users/jacoblopez/Documents/Cheyanne/MS in Data Science/AI Agents/rag_eval/results/eval_20260225_150038.json"),
        Path("/Users/jacoblopez/Documents/Cheyanne/MS in Data Science/AI Agents/rag_eval/results/eval_20260225_151931.json"),
        Path("/Users/jacoblopez/Documents/Cheyanne/MS in Data Science/AI Agents/rag_eval/results/eval_20260225_154505.json"),
    ]
    generator.analyze_and_report(
        report_paths=report_paths,
        dataset_name="KB Batch 1",
        dataset_description="Analysis of baseline hybrid search, top_k=5, batch 1, 5 runs",
        threshold=0.5
    )
