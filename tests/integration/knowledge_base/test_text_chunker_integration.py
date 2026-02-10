import pytest

from pathlib import Path
from unittest.mock import patch

class TestSplitText:
    """Integration tests for the main split_text entry point."""

    def test_full_pipeline_with_pdf(self, text_chunker, valid_md_file_content, valid_md_filename):
        """
        Verifies the entire pipeline runs:
        Markdown Header Split -> Recursive Split -> Metadata Enrichment -> Final Output
        """
        text_chunker.chunk_size = 50
        text_chunker.chunk_overlap = 0

        expected_source_name = "Valid Test Document"
        expected_id_prefix = "valid_test_document"

        # Intentionally passing the raw filename to test the function's file cleaning logic.
        docs = text_chunker.split_text(valid_md_file_content, source_name = valid_md_filename)

        # Header splitter should proudce four splits. 
        # Section 1 and Subsection A are greater than 50 characters, thus they will be further split.
        # To verify the recursive splitter, check that the # of docs is > 4. 
        assert len(docs) > 4

        # Check that the pipeline processed the entire md_file.
        assert any("Conclusion" in d.page_content for d in docs)

        subsection_chunks = [d for d in docs if "Subsection A" in d.page_content and "Context:" in d.page_content]
        
        # Section 2 should be further broken down, as the initial chunk length > 50.
        assert len(subsection_chunks) > 1

        sample_chunk_1 = subsection_chunks[0]

        # Grab the second chunk
        sample_chunk_2 = subsection_chunks[1]
        # Check to make sure that the header hierarchy is prepended in every chunk's content.
        assert "Headers: Title > Section 1 > Subsection A" in sample_chunk_1.page_content
        assert "Headers: Title > Section 1 > Subsection A" in sample_chunk_2.page_content
        
        # Verify source was added to the metadata and to the content.
        assert f"Source: Document Library / {expected_source_name}" in sample_chunk_2.page_content
        assert sample_chunk_2.metadata['source'] == expected_source_name
        
        # Verify that the id is assembled correctly
        sample_chunk_1_id = sample_chunk_1.metadata['id']
        sample_chunk_2_id = sample_chunk_2.metadata['id']
        assert f"{expected_id_prefix}_chunk" in sample_chunk_2_id
        # Verify that the chunks do not have the same id
        assert sample_chunk_1_id != sample_chunk_2_id
    def test_full_pipeline_with_yaml(self, text_chunker):
        """
        Verifies that YAML metadata takes precedence over filenames.
        Also verifies that content is correctly split by headers even when YAML is present.
        """
        text_chunker.chunk_size = 2000
        text_chunker.chunk_overlap = 0
        
        yaml_content = """---
title: Engineering Standards
parent: Engineering Department
path: UWF Public Knowledge Base / Engineering Department / Engineering Standards
page_id: 99999
---
# Safety
Always wear a helmet.

## Equipment
Check your boots.
"""
        # Filename should be overidden
        docs = text_chunker.split_text(yaml_content, source_name="ignored_filename.md")

        assert len(docs) == 2 # Splits on headers first, then on chars. Very small text - only two sections

        safety_chunk = docs[0]
        assert "Always wear a helmet" in safety_chunk.page_content
        assert safety_chunk.metadata['id'].startswith("99999_chunk_1")
        assert safety_chunk.metadata['source'] == "Engineering Standards"
        assert "Headers: Safety" in safety_chunk.page_content
        assert "Equipment" not in safety_chunk.metadata.get('Header 2', '')
        assert "Source: UWF Public Knowledge Base / Engineering Department / Engineering Standards" in safety_chunk.page_content

        equip_chunk = docs[1]
        assert "Check your boots" in equip_chunk.page_content
        assert equip_chunk.metadata['id'].startswith("99999_chunk_2")
        # Verify Hierarchy (H1 > H2)
        assert "Headers: Safety > Equipment" in equip_chunk.page_content
        assert "Source: UWF Public Knowledge Base / Engineering Department / Engineering Standards" in safety_chunk.page_content
    def test_split_text_handles_errors(self, text_chunker, caplog):
            """
            Verifies that if an underlying library fails:
            1. The error is logged (checking side effects).
            2. The error is re-raised (checking control flow).
            """
            with patch('frontmatter.loads', side_effect=ValueError("Critical Failure")):
                
                with pytest.raises(ValueError, match="Critical Failure"):
                    text_chunker.split_text("some text")

            assert "Error during text chunking" in caplog.text
            assert "Critical Failure" in caplog.text