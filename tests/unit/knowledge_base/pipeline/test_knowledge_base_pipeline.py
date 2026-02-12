import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import logging # Used by caplog
import json

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
        @pytest.mark.parametrize("source_type, source_dir", [
            (SourceType.PDF, "raw_data_path"),
            (SourceType.MARKDOWN, "processed_data_path")
        ])
        def test_successful_pipeline(self, mock_full_KB_pipeline, valid_pdf_filepath, valid_md_filepath, example_test_chunk_from_handbook, source_type, source_dir):
            """
            Verifies the full end-to-end happy path of the run() orchestrator for both source types.

            This test is parameterized to cover both PDF and MARKDOWN ingestion flows,
            ensuring the pipeline correctly routes to the appropriate source directory
            and executes every stage in order.

            Scenario:
                - PDF: Discovers files in raw_data_path, converts to markdown, chunks, exports,
                  embeds, and upserts.
                - MARKDOWN: Discovers files in processed_data_path, skips conversion, chunks,
                  exports, embeds, and upserts.

            Expectation:
                - _discover_files is called with the correct source directory for the given type.
                - _export_chunks is called with the generated chunks.
                - _embed_and_upsert is called with the generated chunks.
                - The method returns the list of chunks produced by the chunker.
            """
            # Arrange
            if source_type == SourceType.PDF:
                mock_full_KB_pipeline._discover_files = MagicMock(return_value = [valid_pdf_filepath])
            elif source_type == SourceType.MARKDOWN:
                mock_full_KB_pipeline._discover_files = MagicMock(return_value = [valid_md_filepath])

            mock_full_KB_pipeline._ensure_markdown_exists = MagicMock(return_value = valid_md_filepath)
            with patch("builtins.open", new_callable = MagicMock) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "Context: Source: Graduate..."
            
                mock_full_KB_pipeline.chunker.split_text = MagicMock(return_value = example_test_chunk_from_handbook)
                mock_full_KB_pipeline._export_chunks = MagicMock()
                mock_full_KB_pipeline._embed_and_upsert = MagicMock()
                # Act
                results = mock_full_KB_pipeline.run(
                    source_type = source_type,
                    specific_files = None
                )
            
            source_dir = getattr(mock_full_KB_pipeline, source_dir)
            
            mock_full_KB_pipeline._discover_files.assert_called_once_with(
                source_dir = source_dir,
                source_type = source_type,
                specific_files = None
            )
            mock_full_KB_pipeline._export_chunks.assert_called_once_with(example_test_chunk_from_handbook)
            mock_full_KB_pipeline._embed_and_upsert.assert_called_once_with(example_test_chunk_from_handbook)
            assert results == example_test_chunk_from_handbook
        def test_unsupported_source_type(self, mock_full_KB_pipeline):
            """
            Verifies that the pipeline raises a ValueError when an invalid source type is provided.

            Scenario:
                - Input: A string "URL" that is not a valid SourceType enum member.

            Expectation:
                - A ValueError is raised before any file discovery occurs.
                - The error message contains "Unsupported SourceType".
                - _discover_files is never called.
            """
            mock_full_KB_pipeline._discover_files = MagicMock()
            with pytest.raises(ValueError) as excinfo:
                mock_full_KB_pipeline.run(
                    source_type = "URL",
                    specific_files = None
                )
            
            assert "Unsupported SourceType" in str(excinfo.value)
            mock_full_KB_pipeline._discover_files.assert_not_called()

        @pytest.mark.parametrize("error_type, error_msg", [
            (ValueError, "No files found"), 
            (FileNotFoundError, "Directory does not exist"),
            (PermissionError, "Access denied"),
        ])
        def test_file_discovery_error(self, mock_full_KB_pipeline, error_msg, error_type):
            """
            Ensures that the run method re-raises expected discovery errors (Fail Fast).
            
            Verifies that when _discover_files raises a handled exception, the pipeline:
            1. Logs an error message for observability.
            2. Re-raises the exception to stop the pipeline immediately.
            """
            mock_full_KB_pipeline._ensure_markdown_exists = MagicMock()
            
            mock_full_KB_pipeline._discover_files = MagicMock(side_effect = error_type(error_msg))

            with pytest.raises(error_type) as excinfo:
                results = mock_full_KB_pipeline.run(
                    source_type = SourceType.PDF,
                    specific_files = None
                )
            
            assert error_msg in str(excinfo.value)
            mock_full_KB_pipeline._ensure_markdown_exists.assert_not_called()
        def test_no_text_chunk_error(self, mock_full_KB_pipeline, valid_pdf_filepath, valid_md_filepath):
            """
            Verifies that the pipeline raises a ValueError when chunking produces no results.

            Scenario:
                - Input: A valid PDF file that is successfully discovered and converted to markdown.
                - Condition: The chunker returns an empty list (e.g., the file content was
                  too short or unparseable).

            Expectation:
                - A ValueError is raised with the message "Pipeline finished with no chunks generated."
                - The pipeline does not proceed to export or embed/upsert stages.
            """
            mock_full_KB_pipeline._discover_files = MagicMock(return_value = [valid_pdf_filepath])
            
            mock_full_KB_pipeline._ensure_markdown_exists = MagicMock(return_value = valid_md_filepath)

            with patch("builtins.open", new_callable = MagicMock) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "Empty Content"

                mock_full_KB_pipeline.chunker.split_text.return_value = []

                with pytest.raises(ValueError, match = "Pipeline finished with no chunks generated.") as excinfo:
                    mock_full_KB_pipeline.run(
                        source_type = SourceType.PDF,
                        specific_files = None
                    )

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
    class TestExportChunks:
        @pytest.mark.parametrize("scenario, passed_filename", [
            ("default", "all_chunks.json"),
            ("custom filename", "special_filename.json"),
        ])
        def test_export_chunks_success(self, mock_full_KB_pipeline, passed_filename, scenario, example_test_chunk_from_handbook):
            """
            Verifies that the chunk export functionality correctly writes processed data to a JSON file.

            This test covers both default and custom filename scenarios to ensure the internal 
            export logic respects optional parameters and interacts correctly with the file system.

            Scenario:
                - Default: Chunks are exported without providing a filename, defaulting to "all_chunks.json".
                - Custom filename: Chunks are exported with a specific filename (e.g., "special_filename.json").

            Input:
                - chunks: A list containing a mock Document object representing a chunk from the 
                "Graduate Student Handbook 2024 2025".
                - passed_filename: The filename string determined by the pytest parameterization.

            Expectation:
                - A JSON file is created at the expected output path within the processed data directory.
                - The exported JSON content contains the correct number of chunks (1).
                - The metadata (e.g., source: "Graduate Student Handbook 2024 2025", id: "...chunk_51") 
                is preserved accurately.
                - The "page_content" from the Document object is correctly mapped to the "content" 
                key in the resulting JSON.
            """
            expected_output_path = mock_full_KB_pipeline.processed_data_path / passed_filename

            if scenario == "custom filename":
                mock_full_KB_pipeline._export_chunks(
                    chunks = example_test_chunk_from_handbook,
                    filename = passed_filename
                )
            else:
                mock_full_KB_pipeline._export_chunks(
                    chunks = example_test_chunk_from_handbook
                )

            # Assert file exists
            assert expected_output_path.exists()

            # Verify the content
            content = expected_output_path.read_text(encoding = "utf-8")
            
            data = json.loads(content)
            
            assert len(data) == 1
            assert data[0]["metadata"]["source"] == "Graduate Student Handbook 2024 2025"
            assert data[0]["metadata"]["id"] == "graduate_student_handbook_2024_2025_chunk_51"
            # Export function saves page_content in "content" key 
            assert "Context: Source: Graduate" in data[0]["content"]
        def test_export_chunks_handles_exception(self, mock_full_KB_pipeline,example_test_chunk_from_handbook, caplog):
            """
            Verifies that the export function gracefully handles I/O exceptions (e.g., PermissionError) 
            without crashing the pipeline.

            This test simulates a failure during the file writing process to ensure robust error 
            handling and appropriate logging.

            Scenario:
                - Input: A valid list of document chunks.
                - Condition: An OS-level 'PermissionError' ("Access Denied") is raised when 
                attempting to open the destination file for writing.

            Expectation:
                - The function catches the exception internally and returns None.
                - A failure message is recorded in the logs (via caplog) containing 
                "Failed to export chunks".
                - The specific error message from the exception ("Access Denied") is present in 
                the log output.
            """
            with patch("builtins.open", side_effect=PermissionError("Access Denied")):
                results = mock_full_KB_pipeline._export_chunks(
                    chunks=example_test_chunk_from_handbook
                )
            
            assert results is None
            assert "Failed to export chunks" in caplog.text
            assert "Access Denied" in caplog.text
    @patch('knowledge_base.pipeline.knowledge_base_pipeline.create_vector_db_index')
    @patch('knowledge_base.pipeline.knowledge_base_pipeline.upsert_to_vector_db')
    class TestEmbedandUpsert:
        @pytest.fixture
        def mock_KB_pipeline_with_return_values(self, mock_full_KB_pipeline, normalized_dense_embeddings, normalized_sparse_embeddings):
            mock_full_KB_pipeline.pc.has_index.return_value = True
            mock_full_KB_pipeline.gemini_embedder.embed_KB_document_dense.return_value = normalized_dense_embeddings
            mock_full_KB_pipeline.pinecone_embedder.embed_KB_document_sparse.return_value = normalized_sparse_embeddings

            yield mock_full_KB_pipeline

        def test_embed_and_upsert_happy_path(self, mock_upsert_to_vector_db, mock_create_vector_db_index,mock_KB_pipeline_with_return_values, normalized_dense_embeddings, normalized_sparse_embeddings, example_test_chunk_from_handbook, caplog):
            """
            Verifies the full happy path of _embed_and_upsert when the index already exists.

            Scenario:
                - Input: A valid list of document chunks.
                - Condition: The Pinecone index already exists (has_index returns True).

            Expectation:
                - create_vector_db_index is NOT called (index already exists).
                - The "already exists" log message is recorded.
                - Dense embeddings are generated via Gemini with the correct chunks.
                - Sparse embeddings are generated via Pinecone with the correct chunks.
                - upsert_to_vector_db is called with the correct client, index name,
                  chunks, and both embedding types.
            """
            # Index exists - Not created
            kb_name = mock_KB_pipeline_with_return_values.kb_name

            results = mock_KB_pipeline_with_return_values._embed_and_upsert(
                chunks = example_test_chunk_from_handbook
            )

            assert results is None

            assert f"Pinecone index '{kb_name}' already exists." in caplog.text
            
            mock_create_vector_db_index.assert_not_called()
            
            mock_KB_pipeline_with_return_values.gemini_embedder.embed_KB_document_dense.assert_called_once_with(document = example_test_chunk_from_handbook)
            mock_KB_pipeline_with_return_values.pinecone_embedder.embed_KB_document_sparse.assert_called_once_with(inputs = example_test_chunk_from_handbook)
            
            mock_upsert_to_vector_db.assert_called_once_with(
                pinecone_client = mock_KB_pipeline_with_return_values.pc,
                index_name = kb_name,
                text_chunks = example_test_chunk_from_handbook,
                dense_embeddings = normalized_dense_embeddings,
                sparse_embeddings = normalized_sparse_embeddings
            )
        def test_index_not_exist(self,mock_upsert_to_vector_db, mock_create_vector_db_index, mock_KB_pipeline_with_return_values, example_test_chunk_from_handbook, normalized_dense_embeddings, normalized_sparse_embeddings):
            """
            Verifies that when the Pinecone index does not exist, the pipeline creates it
            before proceeding with embeddings and upserting.

            Scenario:
                - Input: A valid list of document chunks.
                - Condition: has_index returns False.

            Expectation:
                - create_vector_db_index is called with the correct Pinecone client and index name.
                - The pipeline continues to embed and upsert successfully after index creation.
            """
            mock_KB_pipeline_with_return_values.pc.has_index.return_value = False

            mock_KB_pipeline_with_return_values._embed_and_upsert(
                chunks = example_test_chunk_from_handbook
            )

            mock_create_vector_db_index.assert_called_once_with(
                pinecone_client = mock_KB_pipeline_with_return_values.pc,
                index_name = mock_KB_pipeline_with_return_values.kb_name)
  
        def test_index_creation_fails(self,mock_upsert_to_vector_db, mock_create_vector_db_index, mock_KB_pipeline_with_return_values, example_test_chunk_from_handbook, caplog):
            """
            Verifies that when Pinecone index creation fails, the pipeline halts immediately
            without attempting any embedding or upserting operations.

            Scenario:
                - Input: A valid list of document chunks.
                - Condition: has_index returns False, and create_vector_db_index raises an Exception.

            Expectation:
                - The exception propagates to the caller.
                - A "Failed to verify/create Pinecone index" error is logged.
                - upsert_to_vector_db is never called.
            """
            mock_KB_pipeline_with_return_values.pc.has_index.return_value = False
            mock_create_vector_db_index.side_effect = Exception("Pinecone index creation failure")

            with pytest.raises(Exception) as excinfo:
                mock_KB_pipeline_with_return_values._embed_and_upsert(
                    chunks = example_test_chunk_from_handbook
                )

            assert "Failed to verify/create Pinecone index" in caplog.text
            assert "Pinecone index creation failure" in str(excinfo.value)
            mock_upsert_to_vector_db.assert_not_called()
        def test_dense_embedding_fails(self,mock_upsert_to_vector_db, mock_create_vector_db_index, mock_KB_pipeline_with_return_values, example_test_chunk_from_handbook, caplog):
            """
            Verifies that when dense embedding generation fails, the pipeline halts before
            attempting sparse embedding or upserting.

            Scenario:
                - Input: A valid list of document chunks.
                - Condition: The Gemini embedder raises a RuntimeError during dense embedding.

            Expectation:
                - The RuntimeError propagates to the caller.
                - A "Dense embedding failed" error is logged at the pipeline level.
                - Sparse embedding is never attempted.
                - upsert_to_vector_db is never called.
            """
            mock_KB_pipeline_with_return_values.gemini_embedder.embed_KB_document_dense.side_effect = RuntimeError("Error generating dense embeddings for batch")

            with pytest.raises(RuntimeError) as excinfo:
                mock_KB_pipeline_with_return_values._embed_and_upsert(
                    chunks = example_test_chunk_from_handbook
                )
            
            assert "Dense embedding failed" in caplog.text
            mock_KB_pipeline_with_return_values.pinecone_embedder.embed_KB_document_sparse.assert_not_called()
        def test_sparse_embedding_fails(self,mock_upsert_to_vector_db, mock_create_vector_db_index, mock_KB_pipeline_with_return_values, example_test_chunk_from_handbook, caplog, normalized_dense_embeddings):
            """
            Verifies that when sparse embedding generation fails, the pipeline halts before
            attempting to upsert vectors.

            Scenario:
                - Input: A valid list of document chunks.
                - Condition: Dense embedding succeeds, but the Pinecone sparse embedder
                  raises a RuntimeError.

            Expectation:
                - The RuntimeError propagates to the caller.
                - A "Sparse embedding failed" error is logged at the pipeline level.
                - upsert_to_vector_db is never called.
            """
            mock_KB_pipeline_with_return_values.pinecone_embedder.embed_KB_document_sparse.side_effect = RuntimeError("Error generating sparse embeddings")

            with pytest.raises(RuntimeError) as excinfo:
                mock_KB_pipeline_with_return_values._embed_and_upsert(
                    chunks = example_test_chunk_from_handbook
                )

            assert "Sparse embedding failed" in caplog.text
            mock_upsert_to_vector_db.assert_not_called()

        def test_upsert_fails(self,mock_upsert_to_vector_db, mock_create_vector_db_index, mock_KB_pipeline_with_return_values, example_test_chunk_from_handbook, caplog):
            """
            Verifies that when the upsert operation fails, the pipeline logs the error
            and propagates the exception to the caller.

            Scenario:
                - Input: A valid list of document chunks.
                - Condition: Both dense and sparse embeddings succeed, but upsert_to_vector_db
                  raises an Exception.

            Expectation:
                - The Exception propagates to the caller.
                - A "Upsertion failed for KB" error is logged at the pipeline level.
            """
            mock_upsert_to_vector_db.side_effect = Exception("Upsertion failed")

            with pytest.raises(Exception) as excinfo:
                mock_KB_pipeline_with_return_values._embed_and_upsert(
                    chunks = example_test_chunk_from_handbook
                )
            
            assert "Upsertion failed for KB" in caplog.text
            assert "Upsertion failed" in str(excinfo.value)




