import pytest
from langchain_core.documents import Document
from unittest.mock import patch
from pathlib import Path
from typing import List

class TestTextChunker:
    """
    Unit tests for the TextChunker class in knowledge_base.processing.text_chunker.
    """
    class TestSplitOnHeaders:
        """Test the _split_on_headers method."""
        def _assert_fallback_behavior(self, docs, original_text):
            """Helper to verify that the function returned the original text as a fallback."""
            assert len(docs) == 1
            # Metadata should be empty since no headers to split on.
            assert docs[0].metadata == {}
            # Page content should be that which was passed in.
            assert docs[0].page_content == original_text

        def test_splits_headers_correctly(self, text_chunker, valid_md_file_content):
            """Verify that markdown text is correctly split into chunks based on header hierarchy."""
            docs = text_chunker._split_on_headers(valid_md_file_content)

            assert len(docs) == 4

            assert docs[0].metadata["Header 1"] == "Title"
            assert "Intro text" in docs[0].page_content
            # Check that the lowest level header is injected as content.
            assert "# Title" in docs[0].page_content

            assert docs[2].metadata["Header 1"] == "Title"
            assert docs[2].metadata["Header 2"] == "Section 1"
            assert docs[2].metadata["Header 3"] == "Subsection A"
            assert "Deep dive" in docs[2].page_content
            # Check that the lowest level header is injected as content.
            assert "Subsection A" in docs[2].page_content

        def test_handles_no_headers(self, text_chunker):
            """Verifies that plain text without markdown headers is returned as a single chunk of text."""
            text = "Plain text.\nWith no markdown headers"

            docs = text_chunker._split_on_headers(text)

            self._assert_fallback_behavior(docs = docs, original_text = text)
        
        def test_exception_header_splits(self, text_chunker, valid_md_file_content):
            """Verify that the method returns the original text if the underlying splitter crashes."""
            path_to_patch = "knowledge_base.processing.text_chunker.MarkdownHeaderTextSplitter"

            # Mock the splitter to crash
            with patch(path_to_patch, side_effect = Exception("Parsing error")):
                    docs = text_chunker._split_on_headers(valid_md_file_content)

            self._assert_fallback_behavior(docs = docs, original_text = valid_md_file_content)  

    class TestSplitRecursive:
        """Test the _split_recursive method."""    
        def test_splits_large_text_with_overlap(self, text_chunker, long_text_chunk):
            """
            Verify that chunks are split correctly based on length.
            
            Splits should look like:
            
            docs[0].page_content = ABCDEFGHIJKLMNOPQRST
            docs[1].page_content = PQRSTUVWXYZ
            """
            text_chunker.chunk_size = 20
            text_chunker.chunk_overlap = 5

            docs = text_chunker._split_recursive(long_text_chunk)
            
            assert len(docs) == 2

            for doc in docs:
                content = doc.page_content
                assert len(content) <= 20
            
            expected_overlap = "PQRST"

            assert docs[0].page_content.endswith(expected_overlap)
            assert docs[1].page_content.startswith(expected_overlap)

        def test_splits_small_text(self, text_chunker, short_text_chunk):
            docs = text_chunker._split_recursive(short_text_chunk)

            assert len(docs) == 1
            assert docs[0].page_content == short_text_chunk[0].page_content         

    class TestEnrichMetadata:
        """Test the _enrich_metadata method."""
        
        # Internal fixtures used by this class only.
        @pytest.fixture
        def doc_1(self):
            return Document(page_content = "Duplicate text", metadata={})
        @pytest.fixture
        def doc_2(self):
            return Document(page_content = "Duplicate text", metadata={})

        def _assert_id_behavior(self, docs: List[Document]):
            id_1 = docs[0].metadata['id']
            id_2 = docs[1].metadata['id']

            assert id_1 != id_2

            return (id_1, id_2)

        def test_injects_header_hierarchy_into_content(self, text_chunker, valid_pdf_filepath, short_text_chunk):
            """
            Verify that header metadata is correctly formatted and prepended to the text content.
            """
            original_text = short_text_chunk[0].page_content

            docs = text_chunker._enrich_metadata(documents = short_text_chunk, source_name = valid_pdf_filepath)

            # Check that the headers have been inserted
            assert "First Header > Second Header" in docs[0].page_content
            # Check that the original content is still there
            assert original_text in docs[0].page_content
            # Check that "Context: is added at the beginning"
            assert docs[0].page_content.startswith("Context:")

        def test_adds_source_metadata_and_content(self, text_chunker, valid_pdf_filepath, short_text_chunk):
            """
            Verify that source filenames are cleaned, added to metadata, and injected into content text.
            """
            docs = text_chunker._enrich_metadata(
                documents = short_text_chunk, 
                source_name = valid_pdf_filepath)

            # Replicate function cleaning logic
            file_name = Path(valid_pdf_filepath).stem
            expected_source_name = file_name.replace("-", " ").replace("_", " ")
            expected_id_prefix = file_name.replace(" ","_").replace("-","_").lower()

            # Verify metadata
            assert 'id' in docs[0].metadata
            assert expected_id_prefix in docs[0].metadata['id']
            assert "chunk" in docs[0].metadata['id']

            assert 'source' in docs[0].metadata
            assert expected_source_name == docs[0].metadata['source']

            # Verify content injection
            doc_content = docs[0].page_content
            assert f"Source: {expected_source_name}" in doc_content

        def test_assigns_unique_ids_with_source(self, text_chunker, valid_pdf_filepath, doc_1, doc_2):
            """
            Verify IDs use the filename and an incrementing counter when source is provided.
            """
            docs = text_chunker._enrich_metadata([doc_1, doc_2], source_name = valid_pdf_filepath)

            id_1, id_2 = self._assert_id_behavior(docs)

            assert "chunk_1" in id_1
            assert "chunk_2" in id_2
        
        def test_assigns_unique_ids_without_source(self, text_chunker, doc_1, doc_2):
            """
            Verify IDs are unique even when source is missing and content is identical.
            This ensures the hashing logic includes the chunk index to prevent collisions.
            """

            docs = text_chunker._enrich_metadata([doc_1, doc_2])

            id_1, id_2 = self._assert_id_behavior(docs)

            assert id_1.startswith("anon_")
            assert id_2.startswith("anon_")

            # Length should be 37 characters (32 char hash + 5 char prefix)
            assert len(id_1) == 37
            assert len(id_2) == 37

        
        def test_handles_missing_metadata_and_source(self, text_chunker):
            """
            Test case for missing both header metadata and source information.
            """
            plain_text = "Plain text."
            plain_document = Document(
                page_content = plain_text,
                metadata = {}
            )

            docs = text_chunker._enrich_metadata(documents=[plain_document])

            assert len(docs) == 1

            doc_content = docs[0].page_content
            # Check that the original content was kept.
            assert plain_text in doc_content
            # Check that the function didn't unnecessarily add any characters related to the metadata.
            assert ">" not in docs[0].page_content
            assert "Context:" not in doc_content

            # Check that 'source' was not added into the metadata.
            assert 'source' not in docs[0].metadata

            # Check that 'id' was generated correctly:
            assert 'id' in docs[0].metadata
            assert 'anon_' in docs[0].metadata['id']
        def test_handles_missing_source_name_gracefully(self, text_chunker, short_text_chunk):
            """
            Verify that headers are still injected even if source_name is missing.
            """
            docs = text_chunker._enrich_metadata(documents = short_text_chunk)

            doc_content = docs[0].page_content

            assert 'Source' not in doc_content
            assert doc_content.startswith("Context:")
            assert '>' in doc_content
            assert 'id' in docs[0].metadata
            assert 'anon_' in docs[0].metadata['id']