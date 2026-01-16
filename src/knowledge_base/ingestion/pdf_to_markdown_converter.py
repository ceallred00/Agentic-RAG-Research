import logging
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_core.documents import Document
from constants import RAW_DATA_DIR, PROCESSED_DATA_DIR
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)
#TODO: Add logging statements here.

class PDFToMarkdownConverter:
    """
    This class handles the loading of PDF documents and converting them into markdown format.
    It utilizes the DoclingLoader for robust PDF parsing.
    """
    def __init__(self, raw_data_path: Union[str, Path] = RAW_DATA_DIR, processed_data_path: Union[str, Path] = PROCESSED_DATA_DIR):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
    def load_pdf_as_markdown(self, filename: str) -> List[Document]:
        """Loads the file contained at the raw_data_path, appended with the file name. 
         
          Returns a List[Document] object.  """
        import_file_path = self.raw_data_path / filename
        loader = DoclingLoader(file_path=str(import_file_path), 
                               export_type = ExportType.MARKDOWN)

        return loader.load()
    def save_markdown_file(self, content: str, original_filename: str):
        """Strips the original stem from the original filename and replaces it with .md.
         
          Saves the file at the processed_data_path."""
        clean_name = Path(original_filename).stem
        new_filename = f"{clean_name}.md"
        processed_data_dir = Path(self.processed_data_path)
        output_path = processed_data_dir / new_filename

        with open(output_path, "w", encoding = "utf-8") as f:
            f.write(content)

        print(f"Markdown file saved to: {output_path}")

# --- Example Usage ---
if __name__ == "__main__":
    file_name = "AFH1.pdf"

    processor = PDFToMarkdownConverter()

    handbook = processor.load_pdf_as_markdown(file_name)
    if handbook: 
        processor.save_markdown_file(handbook[0].page_content, file_name)
    
    
    
