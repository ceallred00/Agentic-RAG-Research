import logging
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_core.documents import Document
from constants import RAW_DATA_DIR, PROCESSED_DATA_DIR
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

class PDFToMarkdownConverter:
    """
    This class handles the loading of PDF documents and converting them into markdown format.
    It utilizes the DoclingLoader for robust PDF parsing.
    """
    def __init__(self, raw_data_path: Union[str, Path] = RAW_DATA_DIR, processed_data_path: Union[str, Path] = PROCESSED_DATA_DIR):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
    def load_pdf_as_markdown(self, filename: str) -> List[Document]:
        """
        Loads the file contained at the raw_data_path.
        
        Args:
            filename (str): The name of the PDF file (e.g., 'doc.pdf')
            
        Returns:
            List[Document]: A list of LangChain Documents containing the markdown content.
            
        Raises:
            FileNotFoundError: If the source file does not exist.
            RuntimeError: If Docling fails to parse the file.
        """
        import_file_path = self.raw_data_path / filename

        if not import_file_path.exists():
            error_msg = f"File {import_file_path} does not exist."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            loader = DoclingLoader(file_path=str(import_file_path), 
                                export_type = ExportType.MARKDOWN)
            
            documents =  loader.load()
            logger.info(f"Loading PDF file: {import_file_path}")
            return documents
        
        except Exception as e:
            error_msg = f"Failed to load PDF file {import_file_path}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        
    def save_markdown_file(self, content: str, output_filename: str):
        """
        Saves the provided markdown content to a file.
        
        Args:
            content (str): The markdown content to save.
            output_filename (str): The exact filename (including .md extension) to save the content as."""
        try:
            if not self.processed_data_path.exists():
                logger.warning(f"Processed data path {self.processed_data_path} does not exist. Creating it.")
                self.processed_data_path.mkdir(parents=True, exist_ok=True)
            
            output_path = self.processed_data_path / output_filename

            logger.info(f"Saving markdown file to: {output_path}")

            with open(output_path, "w", encoding = "utf-8") as f:
                f.write(content)

            logger.info(f"Markdown file saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save markdown file {output_filename}: {e}", exc_info=True)
            raise


# --- Example Usage ---
if __name__ == "__main__":
    file_name = "AFH1.pdf"

    processor = PDFToMarkdownConverter()

    handbook = processor.load_pdf_as_markdown(file_name)
    if handbook: 
        processor.save_markdown_file(handbook[0].page_content, file_name)
    
    
    
