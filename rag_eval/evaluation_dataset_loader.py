import csv
import logging

from pathlib import Path
from typing import List, Union, Optional
from pydantic import ValidationError
from rag_eval.schemas.eval_schemas import EvalDatasetRow


from constants import RAG_EVAL_DATA_DIR


logger = logging.getLogger(__name__)

class EvaluationDatasetLoader:
    """
    Loads and validates evaluation datasets from CSV files.

    Uses a two-step validation process:
    1. File-level validation (existence, extension, headers) via _validate_file.
    2. Row-level validation (Pydantic schema) via _parse_row.

    Args:
        csv_dir: Directory containing evaluation CSV files. Defaults to RAG_EVAL_DATA_DIR.
        encoding: File encoding for reading CSVs. Defaults to 'utf-8'.
    """
    def __init__(self, csv_dir: Union[str, Path] = RAG_EVAL_DATA_DIR, encoding: str = "utf-8"):
        self.csv_dir = Path(csv_dir)
        self.encoding = encoding
    def load_eval_dataset(self,csv_filename: str) -> List[EvalDatasetRow]:
        """
        Orchestrates file validation and row-by-row parsing of an evaluation CSV.

        Args:
            csv_filename: Name of the CSV file (e.g., 'eval_dataset.csv') located in self.csv_dir.

        Returns:
            List of validated EvalDatasetRow objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no valid rows are found after parsing.
            RuntimeError: If a file I/O or encoding error occurs.
        """
        file_path = self.csv_dir / csv_filename

        rows = self._validate_file(csv_filepath = file_path)

        validated_rows = []
        for index, row in enumerate(rows):
            val_row = self._parse_row(raw_row = row, row_index = index)
            if val_row is not None:
                validated_rows.append(val_row)
        if not validated_rows:
            error_msg = f"No valid rows found in {file_path}. Verify format against Pydantic model."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return validated_rows
    def _validate_file(self, csv_filepath: Path) -> List[dict]:
        """
        Validates the CSV file and reads its contents.

        Performs the following checks in order:
        1. File existence.
        2. File extension (.csv).
        3. Header validation against EvalDatasetRow model fields.
        4. Non-empty row content.

        Args:
            csv_filepath: Full path to the CSV file.

        Returns:
            List of row dicts (keys are column headers, values are cell contents).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .csv, headers don't match, or file has no data rows.
            RuntimeError: If an OS-level or encoding error occurs during file reading.
        """
        logger.info(f"Validating {csv_filepath} exists and is a .csv file.")
        if not csv_filepath.exists():
            error_msg = f"File not found: {csv_filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        suffix = csv_filepath.suffix
        if suffix != ".csv":
            error_msg = f"Unsupported file type: {suffix}. Please upload a CSV file."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"{csv_filepath} validated.")

        rows = []

        try:
            with open(csv_filepath, newline = '', mode='r', encoding = self.encoding) as file:
                logger.info(f"Opening file: {csv_filepath}")
                # Uses first row's values as headers, rows returned as dictionaries
                reader = csv.DictReader(file)

                # Grab the header row
                headers = reader.fieldnames
                expected_headers = list(EvalDatasetRow.model_fields.keys())

                # Verify headers
                if headers != expected_headers:
                    error_msg = f"Incorrect header format. Please verify against expected headers: {expected_headers}."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info("Verified headers")

                for row in reader:
                    # Append dict to list
                    rows.append(row)

                if not rows:
                    error_msg = f"{csv_filepath} is empty. Please verify content."
                    logger.error(error_msg)
                    raise ValueError(error_msg)            
                else:
                    logger.info(f"Returning rows from {csv_filepath}")
                    return rows

        except (OSError, UnicodeDecodeError) as e:
            raise RuntimeError(f"Error reading CSV file: {e}")

    def _parse_row(self, raw_row: dict, row_index:int) -> Optional[EvalDatasetRow]:
        """
        Validates a single CSV row against the EvalDatasetRow Pydantic model.

        Args:
            raw_row: Dict of column header to cell value (from csv.DictReader).
            row_index: Zero-based row index for logging purposes.

        Returns:
            EvalDatasetRow if validation succeeds, None if validation fails.
        """
        try:
            row = EvalDatasetRow(**raw_row)
            return row
        except ValidationError as e:
            logger.warning(f"Skipping row {row_index}: {e}")
            return None

if __name__ == "__main__": # pragma: no cover
    loader = EvaluationDatasetLoader(csv_dir = RAG_EVAL_DATA_DIR)

    file_name = "KB_testing_dataset.csv"

    dataset = loader.load_eval_dataset(file_name)
    print(f"Returned {len(dataset)} rows.")

    for data_row in dataset[:10]:
        print(data_row)
