import logging
import re
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)


class FileSaver:
    """
    Persists processed data to the local file system.

    Attributes:
        processed_data_path (Path): The directory where processed data will be saved.
    """

    def __init__(self, processed_data_path: Union[str, Path]):
        self.processed_data_path = Path(processed_data_path)

    def save_markdown_file(self, content: str, output_filename: str):
        """
        Saves markdown content to a sanitized, ASCII-safe file path.

        This method ensures the output integrity by:
        1. Verifying the output directory exists (creating it if necessary).
        2. Stripping leading and trailing whitespace from the filename.
        3. Replacing all non-alphanumeric characters (including hyphens, pipes, and slashes) with underscores.
        4. Collapsing multiple underscores into a single delimiter.
        5. Enforcing a '.md' extension on the final path.

        Args:
            content (str): The markdown-formatted string to be saved.
            output_filename (str): The desired name of the file (e.g., page title).

        Example:
            >>> saver.save_markdown_file(content, "Textbook Adoption - Follett | Discover (Faculty / Adopter)")
            # Saves to: Textbook_Adoption_Follett_Discover_Faculty_Adopter.md

        Raises:
            IOError: If the file cannot be written to the disk.
        """
        try:
            if not self.processed_data_path.exists():
                logger.warning(f"Processed data path {self.processed_data_path} does not exist. Creating it.")
                self.processed_data_path.mkdir(parents=True, exist_ok=True)

            if not output_filename:
                logger.warning(f"Cannot save a file with no name... Check upstream functions")
                return

            # Prevents file names which may crash the system
            clean_filename = re.sub(
                r"[^a-zA-Z0-9]", "_", output_filename.strip()
            )  # Replace special chars with underscore + strip outer whitespace
            clean_filename = re.sub(r"_{2,}", "_", clean_filename).strip(
                "_"
            )  # Collapses multiple underscores into a single underscore, then strips the outer underscores.
            # Adds .md filename if none exists. Replaces existing file type with .md.
            output_path = (self.processed_data_path / clean_filename).with_suffix(".md")

            logger.info(f"Saving markdown file to: {output_path}")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Markdown file saved to: {output_path}")
        except IOError as e:
            logger.error(f"Failed to save markdown file {output_filename}: {e}", exc_info=True)
            raise
