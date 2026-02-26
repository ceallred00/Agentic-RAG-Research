from datetime import datetime
from pathlib import Path
from typing import Union, Tuple, TypeVar, Generic
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging

from constants import RAG_EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)

# T can be any type, as long as it's subclass of Pydantic's BaseModel
T = TypeVar('T', bound=BaseModel)

class ReportGenerator(ABC, Generic[T]):
    """
    Abstract base class for report generators.

    Provides shared infrastructure for writing timestamped JSON and Markdown
    reports from any Pydantic model (T). Subclasses must implement
    _write_markdown to define Markdown report-specific content and formatting.

    Type parameter T must be a subclass of BaseModel.

    Args:
        output_dir: Directory where reports will be written.
                    Defaults to RAG_EVAL_RESULTS_DIR.
        prefix: Filename prefix used to distinguish report types
                (e.g. 'eval' for per-run reports, 'analysis' for cross-run reports).
                Defaults to 'eval'.
    """
    def __init__(self, output_dir: Union[str, Path] = RAG_EVAL_RESULTS_DIR, prefix: str = "eval"):
        self.output_directory = Path(output_dir)
        self._prefix = prefix
    def generate_report(self, report: T)-> Tuple[Path, Path]:
        """
        Orchestrates JSON and Markdown report generation for the given report object.

        Validates the output directory, generates timestamped file paths, writes
        both report formats, and returns their paths.

        Args:
            report: A Pydantic model instance (T) containing the report data.

        Returns:
            Tuple of (json_path, md_path) for the written report files.

        Raises:
            FileNotFoundError: If the output directory does not exist.
            OSError: If writing either report file fails.
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
        """
        Generates timestamped JSON and Markdown file paths in the output directory.

        Uses self._prefix to distinguish report types
        (e.g. 'eval' for per-run reports, 'analysis' for cross-run reports).

        Returns:
            Tuple of (json_path, md_path, timestamp) where timestamp is the
            formatted string used in both filenames.
        """
        logger.info("Generating filepaths...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._prefix}_{timestamp}"
        json_path = self.output_directory / f"{filename}.json"
        md_path = self.output_directory / f"{filename}.md"
        logger.info(f"Two filepaths generated: {json_path, md_path}")
        return json_path, md_path, timestamp
    def _write_JSON(self, report: T, json_output_path: Path):
        """
        Serializes a Pydantic model to a formatted JSON file.

        Uses model_dump_json for serialization, which handles nested models
        and custom field types. Accepts any T (bound to BaseModel), making
        this method reusable across all subclasses without modification.

        Args:
            report: The Pydantic model instance to serialize.
            json_output_path: Full path to the output JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        # Gives a formatted JSON string
        logger.info(f"Generating JSON from {type(report).__name__}")
        try:
            json_report = report.model_dump_json(indent=2)
            json_output_path.write_text(data = json_report, encoding = 'utf-8')
        except OSError as e:
            error_msg = f"Failed to write report to {json_output_path}: {e}"
            logger.error(error_msg)
            raise OSError(error_msg) from e
    @abstractmethod
    def _write_markdown(self, report: T, md_output_path: Path, timestamp: str):
        """
        Writes a human-readable Markdown report to disk.

        Subclasses must implement this method to define report-specific
        content and formatting for their report type.

        Args:
            report: The Pydantic model instance (T) containing the report data.
            md_output_path: Full path to the output Markdown file.
            timestamp: Formatted timestamp string (YYYYMMDD_HHMMSS) used for
                       display in the report header.

        Raises:
            OSError: If the file cannot be written.
        """
        ...