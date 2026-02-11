import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import logging # Used by caplog

from langchain_core.documents import Document

from knowledge_base.pipeline.knowledge_base_pipeline import KnowledgeBasePipeline, SourceType


class TestKnowledgeBasePipeline:
    
    @pytest.fixture
    def hollow_pipeline_instance(self):
        """
        Creates a 'hollow' instance of KnowledgeBasePipeline.
        __init__ is patched so it returns None and does not initialize Pinecone,
        Gemini, ExecutionService, or TextChunker instances.
        """
        with patch.object(KnowledgeBasePipeline, '__init__', return_value=None):
            # Manually set any attributes taht are needed.
            pipeline = KnowledgeBasePipeline(kb_name = "test_kb")
            # We must manually set any attributes that _discover_files might need.
        return pipeline
       
    class TestRunPipeline:
        def test_successful_pipeline(self):
            pass
        def test_unsupported_source_type(self, mock_full_KB_pipeline):
            # Assert ValueError is raised if unsupported source type passed.
            with pytest.raises(ValueError) as excinfo:
                mock_full_KB_pipeline.run(
                    source_type = "URL",
                    specific_files = None
                )
            
            assert "Unsupported SourceType" in str(excinfo.value)
    
    class TestDiscoverFiles:
        @pytest.mark.parametrize("source_type, file_ext", [
            (SourceType.MARKDOWN, ".md"),
            (SourceType.PDF, ".pdf")
        ])
        def test_discover_files(self,valid_data_dir, hollow_pipeline_instance, source_type, file_ext):
            """
            Verifies that the pipeline correctly discovers files of a specific type in a directory.

            This test is parameterized to cover both MARKDOWN (.md) and PDF (.pdf) ingestion.
            It ensures that:
            1. Files matching the `source_type` are successfully found.
            2. Files of different types (e.g., .png) are correctly ignored.

            Args:
                valid_data_dir: Pytest fixture providing a temporary directory.
                hollow_pipeline_instance: Pytest fixture providing a mocked pipeline.
                source_type: The enum type of the source (PDF or MARKDOWN).
                file_ext: The expected file extension for that source type.
            """
            (valid_data_dir / f"doc1{file_ext}").touch()
            (valid_data_dir / f"doc2{file_ext}").touch()
            (valid_data_dir / "image.png").touch()

            files = hollow_pipeline_instance._discover_files(
                source_dir = valid_data_dir,
                source_type = source_type,
                specific_files = None
            )

            assert len(files) == 2
            assert all(f.suffix == file_ext for f in files)
        def test_discover_specific_files(self, valid_raw_data_dir, hollow_pipeline_instance):
            """
            Verifies that the pipeline accepts and processes a specific list of files.

            This ensures that when a user provides the `specific_files` argument, 
            the pipeline only processes those exact files and ignores others in the directory.

            Args:
                valid_raw_data_dir: Pytest fixture providing a temporary directory.
                hollow_pipeline_instance: Pytest fixture providing a mocked pipeline.
            """
            specific_files = ["testFile1.pdf", "testFile2.pdf"]
            
            (valid_raw_data_dir / "testFile1.pdf").touch()
            (valid_raw_data_dir / "testFile2.pdf").touch()
            

            files = hollow_pipeline_instance._discover_files(
                source_dir = valid_raw_data_dir, 
                source_type = SourceType.PDF,
                specific_files = specific_files
            )

            assert len(files) == 2
            assert all(f.suffix == ".pdf" for f in files)
        def test_directory_not_exist(self, hollow_pipeline_instance):
            """
            Verifies that the pipeline raises FileNotFoundError if the source directory is missing.

            The pipeline should fail fast before attempting any scanning if the
            provided `source_dir` path is invalid.
            """
            # Assert the test raises a FileNotFoundError
            with pytest.raises(FileNotFoundError) as excinfo:
                files = hollow_pipeline_instance._discover_files(
                    source_dir = Path("not/valid/directory"),
                    source_type = SourceType.MARKDOWN,
                    specific_files = None
                )
            
            assert "Source directory does not exist" in str(excinfo.value)
        def test_missing_specific_files(self, valid_processed_data_dir, hollow_pipeline_instance):
            """
            Verifies that the pipeline raises FileNotFoundError if requested specific files are missing.

            If a user explicitly asks for `ghost_file.md` but it does not exist on disk,
            the pipeline should stop and report the missing file rather than silently proceeding.
            """
            # Assert the test raises a FileNotFoundError
            with pytest.raises(FileNotFoundError) as excinfo:
                files = hollow_pipeline_instance._discover_files(
                    source_dir = valid_processed_data_dir, 
                    source_type = SourceType.MARKDOWN,
                    specific_files = ["ghost_file.md"]
                )
            
            assert "Failed to find 1 requested files" in str(excinfo.value)
        def test_no_files_found(self, caplog, valid_processed_data_dir, hollow_pipeline_instance):
            """
            Verifies that the pipeline raises ValueError if the directory is empty or contains no matching files.

            This prevents the pipeline from running unnecessary downstream tasks (like embedding)
            when there is no input data to process.
            """
            # Assert the test raises a ValueError
            with pytest.raises(ValueError) as excinfo:
                files = hollow_pipeline_instance._discover_files(
                    source_dir = valid_processed_data_dir, 
                    source_type = SourceType.MARKDOWN,
                    specific_files = None
                )
            
            assert "No MARKDOWN files found" in str(excinfo.value)
        def test_permission_error(self, caplog, hollow_pipeline_instance, valid_processed_data_dir):
            """
            Verifies that permission errors are logged and re-raised during file discovery.

            This uses mocking to simulate a system-level PermissionError (e.g., chmod 000)
            without needing to actually manipulate filesystem permissions in the test environment.
            It checks that:
            1. The error is logged to the application logs.
            2. The exception is re-raised to stop execution.
            """
            with patch('pathlib.Path.glob', side_effect = PermissionError("Permission Denied")):
                with pytest.raises(PermissionError) as excinfo:
                    hollow_pipeline_instance._discover_files(
                        source_dir = valid_processed_data_dir,
                        source_type = SourceType.PDF, 
                        specific_files = None
                    )
                
                assert "Permission denied accessing directory" in caplog.text
                assert "Permission Denied" in str(excinfo.value)
    class TestEnsureMarkdownExists:
        def test_markdown_file_exists(self, mock_full_KB_pipeline, valid_md_filepath):
            """
            Verifies that when a Markdown file is passed as the source, 
            the function immediately returns its path without attempting conversion.

            Scenario:
                - Input: A valid .md file path that exists in the processed directory.
                - SourceType: MARKDOWN.
            
            Expectation:
                - The function should return the exact same path provided.
                - No conversion logic (load_pdf_as_markdown) should be triggered.
            """
            results = mock_full_KB_pipeline._ensure_markdown_exists(
                file_path = valid_md_filepath,
                source_type = SourceType.MARKDOWN
            )

            assert results == valid_md_filepath 
            
        def test_converts_PDF_file(self, mock_full_KB_pipeline, valid_md_filename, valid_pdf_filepath, valid_md_filepath):
            """
            Verifies that when a PDF file is passed, the function detects the need for conversion,
            calls the converter service, and returns the path to the new Markdown file.

            Scenario:
                - Input: A valid .pdf file path in the raw directory.
                - SourceType: PDF.
                - Condition: The corresponding .md file does NOT exist yet.
            
            Expectation:
                - The converter's 'load_pdf_as_markdown' method is called with the PDF filename.
                - The converter's 'save_markdown_file' method is called with the extracted content.
                - The function returns a Path object ending in .md (not .pdf).
            """
            if valid_md_filepath.exists():
                valid_md_filepath.unlink()
            
            mock_doc_1 = MagicMock()
            mock_doc_1.page_content = "Mock Content"

            mock_doc_2 = MagicMock()
            mock_doc_2.page_content = "Mock Content...again"
            # Mock the function to return a mock list of Docs.
            mock_full_KB_pipeline.converter.load_pdf_as_markdown.return_value = [mock_doc_1, mock_doc_2]


            results = mock_full_KB_pipeline._ensure_markdown_exists(
                file_path = valid_pdf_filepath,
                source_type = SourceType.PDF
            )

            expected_page_content = "Mock Content\n\nMock Content...again"

            expected_file_name = Path(mock_full_KB_pipeline.processed_data_path / f"{valid_md_filename}")

            assert results == expected_file_name
            mock_full_KB_pipeline.converter.load_pdf_as_markdown.assert_called_once_with(valid_pdf_filepath.name)
            mock_full_KB_pipeline.converter.save_markdown_file.assert_called_once_with(content = expected_page_content, output_filename = valid_md_filename)
        
        def test_pdf_skips_conversion_if_md_exists(self, mock_full_KB_pipeline, valid_pdf_filepath, valid_md_filepath, caplog):
            """
            Verifies that if the Markdown file already exists for a given PDF,
            the pipeline skips the conversion process and returns the existing path.

            Scenario:
                - Input: A valid .pdf file.
                - Condition: The corresponding .md file ALREADY exists (provided by fixture).
            
            Expectation:
                - The function returns the path to the existing .md file.
                - The converter is NOT called.
            """
            results = mock_full_KB_pipeline._ensure_markdown_exists(
                file_path = valid_pdf_filepath,
                source_type = SourceType.PDF
            )

            assert results == valid_md_filepath
            assert "Skipping conversion" in caplog.text
            mock_full_KB_pipeline.converter.load_pdf_as_markdown.assert_not_called()

        def test_pdf_conversion_handles_exceptions(self, mock_full_KB_pipeline, valid_raw_data_dir, caplog):
            """
            Verifies that the pipeline gracefully handles exceptions raised by the converter 
            (e.g., due to file corruption) by logging the error and returning None, rather than crashing.

            Scenario:
                - Input: A valid path to a PDF file (simulating a corrupt file).
                - SourceType: PDF.
                - Condition: The converter's 'load_pdf_as_markdown' method raises an exception.
            
            Expectation:
                - The exception is caught within the method (does not bubble up).
                - The function returns None.
                - The converter's 'save_markdown_file' method is NOT called.
            """
            pdf_path = valid_raw_data_dir / "corrupt.pdf"
            pdf_path.touch()

            mock_full_KB_pipeline.converter.load_pdf_as_markdown.side_effect = RuntimeError("Docling failed")

            result = mock_full_KB_pipeline._ensure_markdown_exists(
                    file_path = pdf_path,
                    source_type = SourceType.PDF
                )
            
            assert "Docling failed" in caplog.text
            assert "Conversion failed" in caplog.text
            assert result is None
            # Should not reach here
            mock_full_KB_pipeline.converter.save_markdown_file.assert_not_called()
        def test_pdf_conversion_handles_empty_list(self, mock_full_KB_pipeline, valid_pdf_filepath):
            """
            Verifies that the pipeline handles cases where the converter runs successfully 
            but returns no content (e.g., an empty PDF or one with no extractable text), 
            returning None to skip processing.

            Scenario:
                - Input: A valid .pdf file path in the raw directory.
                - SourceType: PDF.
                - Condition: The converter returns an empty list of documents (no text found).
            
            Expectation:
                - The function detects the empty content.
                - The function returns None.
                - The converter's 'save_markdown_file' method is NOT called.
            """
            mock_full_KB_pipeline.converter.load_pdf_as_markdown.return_value = []

            result = mock_full_KB_pipeline._ensure_markdown_exists(
                file_path = valid_pdf_filepath, 
                source_type = SourceType.PDF
            )

            assert result is None
            mock_full_KB_pipeline.converter.save_markdown_file.assert_not_called()
