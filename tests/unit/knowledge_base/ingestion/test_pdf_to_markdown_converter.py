import pytest
from langchain_core.documents import Document
from langchain_docling.loader import ExportType
from unittest.mock import patch

class TestPDFToMarkdownConverter:
    """Unit tests for the PDFToMarkdownConverter class."""
    class TestLoadPDFAsMarkdown:
        """Tests for the load_pdf_as_markdown method."""

        def test_happy_path(self, pdf_converter,mock_docling_loader, valid_pdf_filename, valid_pdf_filepath, valid_md_file_content):
            """
            Tests loading a valid PDF file and converting it to markdown documents.

            Happy path scenario.
            """

            # Configure the mock loader to return sample documents
            mock_docling_loader.return_value.load.return_value = [
                Document(page_content=valid_md_file_content)]

            documents = pdf_converter.load_pdf_as_markdown(valid_pdf_filename)

            assert len(documents) == 1
            assert documents[0].page_content == valid_md_file_content

            mock_docling_loader.assert_called_once_with(
                file_path =str(valid_pdf_filepath),
                export_type = ExportType.MARKDOWN
            )
        def test_file_not_found(self, pdf_converter):
            """Tests behavior when the specified PDF file does not exist."""
            with pytest.raises(FileNotFoundError):
                pdf_converter.load_pdf_as_markdown("non_existent_file.pdf")
        def test_loader_exception(self, pdf_converter, mock_docling_loader, valid_pdf_filename, valid_pdf_filepath):
            """Tests behavior when the DoclingLoader raises an exception."""
            mock_docling_loader.return_value.load.side_effect = Exception("Docling parsing error")
            with pytest.raises(RuntimeError) as exc_info:
                pdf_converter.load_pdf_as_markdown(valid_pdf_filename)
            
            assert "Failed to load PDF file" in str(exc_info.value)
    class TestSaveMarkdownFile:
        """Tests for the save_markdown_file method."""
        
        def test_save_markdown_file_success(self, pdf_converter, valid_processed_data_dir, valid_md_file_content, valid_md_filename):
            """Tests successful saving of markdown content to a file."""
            expected_file_path = valid_processed_data_dir / valid_md_filename

            # Check to make sure the file does not exist before saving
            assert not expected_file_path.exists()

            pdf_converter.save_markdown_file(content = valid_md_file_content, output_filename = valid_md_filename)

            assert expected_file_path.exists()

            assert expected_file_path.read_text() == valid_md_file_content

        def test_save_markdown_file_io_error(self, valid_md_file_content, valid_md_filename, pdf_converter):
            # When calling "open", it will raise an IOError
            with patch("builtins.open", side_effect=IOError("Unable to write file")):
                with pytest.raises(IOError, match = "Unable to write file") as exc_info:
                    pdf_converter.save_markdown_file(content = valid_md_file_content, output_filename = valid_md_filename)
                
                assert "Unable to write file" in str(exc_info.value)
        def test_save_markdown_file_creates_directory(self, pdf_converter, valid_data_dir, valid_md_file_content, valid_md_filename):
            """Tests that the processed data directory is created if it does not exist."""
            fake_processed_dir = valid_data_dir / "fake_processed"
            pdf_converter.processed_data_path = fake_processed_dir

            # Ensure the processed data directory does not exist
            assert not fake_processed_dir.exists()

            pdf_converter.save_markdown_file(content = valid_md_file_content, output_filename = valid_md_filename)

            assert fake_processed_dir.exists()