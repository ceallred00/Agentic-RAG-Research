from typing import Union
from pathlib import Path

def list_json_files_in_directory(results_directory: Union[str, Path]) -> list[Path]:
    """List all JSON files in the specified directory."""
    results_dir = Path(results_directory)
    if not results_dir.is_dir():
        raise NotADirectoryError(f"{results_directory} is not a valid directory.")
    return list(results_dir.glob("*.json"))