import pytest

from pathlib import Path 

class TestEvaluationDatasetLoader:
    """Tests for EvaluationDatasetLoader: file validation, row parsing, and end-to-end loading."""

    class TestLoadEvalDataset:
        """Tests for the load_eval_dataset orchestrator method."""
        def test_load_happy_path(self, rag_dataset_loader, valid_csv_file):
            """Valid CSV with 3 rows returns 3 validated EvalDatasetRow objects."""
            results = rag_dataset_loader.load_eval_dataset("test.csv")

            assert len(results) == 3
            assert results[0].question == "What is the deadline for dropping a class?"
            assert results[2].ground_truth == "The John C. Pace Library is located on the main campus."
        def test_mixed_rows(self, rag_dataset_loader, valid_csv_filepath):
            """CSV with one valid and one invalid row returns only the valid row."""
            valid_csv_filepath.write_text(
                "question,ground_truth\n"
                "What is the deadline for dropping a class?,The last day to drop without a W is August 25th.\n"
                "How do I apply for financial aid?,"
            )

            results = rag_dataset_loader.load_eval_dataset("test.csv")

            assert len(results) == 1
            assert results[0].question == "What is the deadline for dropping a class?"
        def test_all_invalid_rows(self, rag_dataset_loader, valid_csv_filepath, caplog):
            """CSV where all rows fail Pydantic validation raises ValueError."""
            valid_csv_filepath.write_text(
                "question,ground_truth\n"
                "What is the deadline for dropping a class?,"
            )

            with pytest.raises(ValueError):
                rag_dataset_loader.load_eval_dataset("test.csv")
            
            assert f"No valid rows found in {valid_csv_filepath}." in caplog.text

    class TestValidateFile:
        """Tests for _validate_file: existence, extension, header, and empty file checks."""
        def test_file_does_not_exist(self, rag_dataset_loader, valid_data_dir, caplog):
            """Non-existent file path raises FileNotFoundError."""
            file_path = valid_data_dir / "non_existent_file.csv"

            with pytest.raises(FileNotFoundError):
                rag_dataset_loader._validate_file(csv_filepath = file_path)

            assert "File not found" in caplog.text

        def test_wrong_extension_file(self, rag_dataset_loader, valid_data_dir, caplog):
            """File with non-.csv extension raises ValueError."""
            file_path = valid_data_dir / "wrong_file_type.txt"
            file_path.touch()

            with pytest.raises(ValueError):
                rag_dataset_loader._validate_file(csv_filepath = file_path)
            
            assert "Unsupported file type" in caplog.text

        def test_incorrect_headers(self, valid_csv_filepath, rag_dataset_loader, caplog):
            """CSV with headers that don't match EvalDatasetRow fields raises ValueError."""
            valid_csv_filepath.write_text("wrong_col1,wrong_col2\nfoo,bar\n")

            with pytest.raises(ValueError):
                rag_dataset_loader._validate_file(csv_filepath=valid_csv_filepath)
            
            assert "Incorrect header format" in caplog.text
        def test_empty_file(self, valid_csv_filepath, rag_dataset_loader, caplog):
            """CSV with valid headers but no data rows raises ValueError."""
            valid_csv_filepath.write_text("question,ground_truth")

            with pytest.raises(ValueError):
                rag_dataset_loader._validate_file(csv_filepath=valid_csv_filepath)
            
            assert f"{valid_csv_filepath} is empty." in caplog.text

    class TestParseRow:
        """Tests for _parse_row: valid rows, empty strings, and None values."""
        def test_handles_valid_row(self, rag_dataset_loader):
            """Valid row dict returns a populated EvalDatasetRow."""
            row = {"question": "What color is the sky?", "ground_truth": "The sky is blue."}
            result = rag_dataset_loader._parse_row(raw_row = row, row_index = 1)

            assert result is not None
            assert result.question == "What color is the sky?"
        @pytest.mark.parametrize("scenario, row_value, row_index", [
            ("empty string", {"question": "", "ground_truth":"Blue"}, 1),
            ("None value", {"question": "What color is the sky?", "ground_truth": None}, 2)
        ])
        def test_handles_malformed_rows(self, scenario, row_value, row_index, rag_dataset_loader, caplog):
            """Rows with empty strings or None values return None and log a warning."""
            results = rag_dataset_loader._parse_row(raw_row = row_value, row_index = row_index)

            assert results is None
            assert f"Skipping row {row_index}" in caplog.text