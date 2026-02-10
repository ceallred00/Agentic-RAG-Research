import pytest
import copy
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
    SCENARIOS = [
        {
            "id": "File_Happy_Path",
            "inputs": {
                "source": "Graduate-Handbook.pdf",
                "yaml": {}
            },
            "expectations": {
                "id_prefix": "graduate_handbook",               # Cleaned filename
                "meta_source": "Graduate Handbook",             # Cleaned title in metadata
                "content_breadcrumb": "Document Library / Graduate Handbook", 
                "last_updated": None,                           # PDF files not expected to have update date
                "version": None,
                "is_anonymous": False
            }
        },
        {
            "id": "File_Anonymous",
            "inputs": {
                "source": None,                                 # Missing Source
                "yaml": {}
            },
            "expectations": {
                "id_prefix": "anon_",                           # Hash fallback
                "meta_source": "Unknown Document",              # Cleaned title in metadata - Uses default
                "content_breadcrumb": "Unknown Document",       # No parent for file
                "last_updated": None,                           # PDF files not expected to have update date
                "version": None,
                "is_anonymous": True
            }
        },
        {
            "id": "Confluence_Mode",
            "inputs": {
                "source": None,
                "yaml": {
                    'title': 'Advising Syllabus', 
                        'parent': 'Academic Advising', 
                        'path': 'UWF Public Knowledge Base / Academic Advising / Advising Syllabus', # Full path provided by Confluence
                        'original_url': 'https://confluence.uwf.edu/pages/viewpage.action?pageId=42669534',
                        'page_id': 42669534,
                        'version': '34', 
                        'last_updated': '2022-04-12T15:55:51.613Z'
                }
            },
            "expectations": {
                "id_prefix": "42669534",                        # Page ID becomes root
                "meta_source": "Advising Syllabus",
                "content_breadcrumb": "UWF Public Knowledge Base / Academic Advising / Advising Syllabus",
                "last_updated": "2022-04-12",                   # Expecting clean YYYY-MM-DD
                "version": '34',
                "is_anonymous": False
            }
        }
    ]

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda x: x["id"])
    def test_enrichment_logic_master(self, text_chunker, short_text_chunk, scenario):
        """
        Validates ID generation, source injection, and header preservation across all modes.
        
        Duplicate content is passed to ensure the ID counter works even when content hashes collide.
        """
        inputs = scenario["inputs"]
        expects = scenario["expectations"]
        
        # Creates deep copy of first document
        # This guarantees that docs[0] and docs[1] have identical content and metadata but that they are not tied to same object.
        # If the ID logic is broken, these will generate the same ID.
        docs = [copy.deepcopy(short_text_chunk[0]) for _ in range(2)]

        enriched = text_chunker._enrich_metadata(
            docs, 
            yaml_meta=inputs["yaml"], 
            source_name=inputs["source"]
        )

        first_doc = enriched[0]
        second_doc = enriched[1]


        # All scenarios should inject headers if they exist in the input doc
        assert "First Header > Second Header" in first_doc.page_content
        assert "First Header > Second Header" in second_doc.page_content

        assert "Context:" in first_doc.page_content
        assert "Context:" in second_doc.page_content
        
        # Verify the ID matches the expected prefix strategy
        assert first_doc.metadata['id'].startswith(expects["id_prefix"])
        
        # Even though content is identical, IDs must be unique due to chunk index
        assert "chunk_1" in first_doc.metadata['id']
        assert "chunk_2" in second_doc.metadata['id']
        assert first_doc.metadata['id'] != second_doc.metadata['id']

        assert first_doc.metadata.get('source') == expects["meta_source"]
        assert first_doc.metadata.get('version') == expects["version"]
        assert first_doc.metadata.get('last_updated') == expects["last_updated"]

        if expects["is_anonymous"]:
            # Verify we are using the hash logic (13+ chars)
            assert len(first_doc.metadata['id']) > 13 

        # Verify the source is cleaned and injected correctly
        expected_line = f"Source: {expects['content_breadcrumb']}"
        assert expected_line in first_doc.page_content
        assert expected_line in second_doc.page_content

        assert expected_line in first_doc.metadata['breadcrumbs']
        assert expected_line in second_doc.metadata['breadcrumbs']

        if expects["last_updated"]:
            assert f"Last Updated: {expects['last_updated']}" in first_doc.page_content
            assert f"Last Updated: {expects['last_updated']}" in second_doc.page_content
            assert f"Version: {expects['version']}" in first_doc.page_content
            assert f"Version: {expects['version']}" in second_doc.page_content

        else:
            # Verify we didn't inject "None" or empty strings into the text
            assert "Last Updated: None" not in first_doc.page_content
            assert "Last Updated: None" not in second_doc.page_content
            assert "Version:" not in first_doc.page_content
            assert "Version:" not in second_doc.page_content

    def test_handles_empty_metadata_source(self, text_chunker):
        """
        Verifies that 'Headers:' and breadcrumbs are omitted when input metadata is empty.
        
        (The Master test cannot check this because it assumes input headers exist).
        """
        plain_document = Document(page_content="Plain text.", metadata={})

        docs = text_chunker._enrich_metadata(
            documents=[plain_document], 
            yaml_meta={}, 
            source_name=None
        )

        # Since metadata={} was empty, the code should not have added the "Headers:" line
        assert "Headers" not in docs[0].page_content
        assert "Headers" not in docs[0].metadata.get("breadcrumbs", "")
        
        # Check fallbacks
        assert docs[0].metadata['source'] == "Unknown Document"
        assert docs[0].metadata['id'].startswith("anon_")
        assert "Context:" in docs[0].page_content
        assert "Unknown Document" in docs[0].page_content
        assert "Headers" not in docs[0].page_content