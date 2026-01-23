from pathlib import Path

class TestSplitText:
    """Integration tests for the main split_text entry point."""

    def test_full_pipeline_integration(self, text_chunker, valid_md_file_content, valid_md_filename):
        """
        Verifies the entire pipeline runs:
        Markdown Header Split -> Recursive Split -> Metadata Enrichment -> Final Output
        """
        text_chunker.chunk_size = 50
        text_chunker.chunk_overlap = 0

        source_file = Path(valid_md_filename).stem
        expected_source_name = source_file.replace("-", " ").replace("_", " ")
        expected_id_prefix = source_file.replace(" ","_").replace("-","_").lower()

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
        assert "Title > Section 1 > Subsection A" in sample_chunk_1.page_content
        assert "Title > Section 1 > Subsection A" in sample_chunk_2.page_content
        
        # Verify source was added to the metadata and to the content.
        assert f"Source: {expected_source_name}" in sample_chunk_2.page_content
        assert sample_chunk_2.metadata['source'] == expected_source_name
        
        # Verify that the id is assembled correctly
        sample_chunk_1_id = sample_chunk_1.metadata['id']
        sample_chunk_2_id = sample_chunk_2.metadata['id']
        assert f"{expected_id_prefix}_chunk" in sample_chunk_2_id
        # Verify that the chunks do not have the same id
        assert sample_chunk_1_id != sample_chunk_2_id