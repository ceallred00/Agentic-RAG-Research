import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import json

from rag_eval.report_generator import ReportGenerator

@pytest.fixture
def sample_timestamp():
    return "20260218_143052"

class TestReportGenerator:
    """Tests for ReportGenerator: file generation, JSON/Markdown writing, and error handling."""

    class TestGenerateReport:
        """Tests for the generate_report orchestrator method."""
        def test_generate_report_happy_path(self, report_generator_instance, sample_eval_report):
            """Valid EvalReport produces both JSON and Markdown files with correct content."""
            # Act
            json_path, md_path = report_generator_instance.generate_report(report = sample_eval_report)

            # Assert
            assert json_path.suffix == ".json"
            assert md_path.suffix == ".md"

            # Verify files exist
            assert json_path.exists()
            assert md_path.exists()

            # Verify correct content and correct JSON format
            json_content = json_path.read_text()
            data = json.loads(json_content)
            assert data["dataset_name"] == "test_dataset.csv"

            # Verify correct content
            md_content = md_path.read_text()
            assert "# RAG Evaluation Report" in md_content
        def test_generate_report_dir_not_exist(self, sample_eval_report, caplog):
            """Non-existent output directory raises FileNotFoundError."""
            generator = ReportGenerator(output_dir= "invalid/data/dir")

            with pytest.raises(FileNotFoundError):
                generator.generate_report(report = sample_eval_report)
            
            assert "Report output directory not found" in caplog.text

    class TestGenerateFilepaths:
        """Tests for _generate_filepaths: timestamped path generation."""
        def test_generate_correct_filepaths(self, report_generator_instance, valid_data_dir, sample_timestamp):
            """Generates JSON and Markdown paths with correct timestamp-based filenames."""
            expected_json = valid_data_dir / f"eval_{sample_timestamp}.json"
            expected_md = valid_data_dir / f"eval_{sample_timestamp}.md"
            
            # Act
            with patch("rag_eval.report_generator.datetime") as mock_dt:
                mock_dt.now.return_value.strftime.return_value = sample_timestamp
                json_path, md_path, timestamp = report_generator_instance._generate_filepaths()
            
            # Assert
            assert json_path == expected_json
            assert md_path == expected_md
            assert timestamp == sample_timestamp
            
    class TestWriteJSON:
        """Tests for _write_JSON: JSON serialization and file writing."""
        def test_valid_JSON(self, report_generator_instance, sample_eval_report, valid_data_dir):
           """Writes valid JSON file with correct EvalReport fields."""
           output_path = valid_data_dir / "test.json"
           report_generator_instance._write_JSON(report = sample_eval_report, json_output_path = output_path)

           assert output_path.exists()

           content = output_path.read_text()

           json_content = json.loads(content)

           assert json_content["dataset_name"] == "test_dataset.csv"
           assert json_content["description"] == "Baseline hybrid search, top_k=5"

        def test_handles_OS_error(self, report_generator_instance, sample_eval_report, valid_data_dir, caplog):
            """OSError during JSON write is logged and re-raised with file path context."""
            output_path = valid_data_dir / "test.json"

            with patch.object(Path, "write_text", side_effect = OSError("disk full")):
                with pytest.raises(OSError):
                    report_generator_instance._write_JSON(report = sample_eval_report,
                                                              json_output_path = output_path)
            assert f"Failed to write report to {output_path}" in caplog.text
    class TestWriteMarkdown:
        """Tests for _write_markdown: Markdown generation and file writing."""
        def test_valid_md(self, report_generator_instance, valid_data_dir, sample_eval_report, sample_timestamp):
            """Writes Markdown with overview, summary table, per-question results, and contexts."""
            output_path = valid_data_dir / "test.md"

            report_generator_instance._write_markdown(report = sample_eval_report, 
                                                      md_output_path = output_path,
                                                      timestamp = sample_timestamp)
            
            assert output_path.exists()

            content = output_path.read_text()

            # Overview
            assert sample_eval_report.description in content
            assert "test_dataset.csv" in content
            assert sample_timestamp in content
            assert "0.90" in content # Avg context precision
            assert "0.85" in content # Avg context recall

            # Check per-question results
            assert "## Per-Question Results" in content
            assert "What is the deadline for dropping a class?" in content
            assert "How do I apply for financial aid?" in content
            assert "0.92" in content # Question 1 context precision
            assert "0.82" in content # Question 2 context recall

            # Check content has been appended correctly
            assert "The last day to drop without a W is August 25th." in content # Q1, context 1
            assert "Drop/Add period ends on" in content #Q1, context 2

            assert "Submit FAFSA" in content # Q2, context 1
            assert "Financial aid office is located" in content # Q2, context 2
            
        def test_handles_OS_error(self, report_generator_instance, sample_eval_report, valid_data_dir,sample_timestamp, caplog):
            """OSError during Markdown write is logged and re-raised with file path context."""
            output_path = valid_data_dir / "test.md"

            # Assert/Act
            with patch.object(Path, "write_text", side_effect = OSError("disk full")):
                with pytest.raises(OSError):
                    report_generator_instance._write_markdown(report = sample_eval_report, 
                                                            md_output_path = output_path, 
                                                            timestamp = sample_timestamp)
            
            assert f"Failed to write report to {output_path}" in caplog.text
