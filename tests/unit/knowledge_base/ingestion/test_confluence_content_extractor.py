from unittest.mock import patch


class TestConfluenceContentExtractor:
    class TestExtract:
        def test_extract_happy_path(self, confluence_html_extractor, sample_url, sample_space_key):
            """
            Verifies the standard 'happy path' conversion of Confluence HTML to Markdown.
            
            This test ensures that core content elements (headers, paragraphs) are 
            preserved and formatted correctly, while invisible artifacts (like 
            non-breaking spaces '\xa0' or empty paragraph tags) are normalized or 
            stripped during the conversion process.
            """
            raw_html = """<h1>Overview</h1><p>This page contains information about academic advising.\xa0</p>
                        <h1>Pages on this Topic</h1><p><br /></p>"""
            
            result = confluence_html_extractor.extract(raw_html = raw_html, 
                                                       base_url = sample_url, 
                                                       space_key = sample_space_key)
            
            expected_md_result = "# Overview\n\nThis page contains information about academic advising.\n\n# Pages on this Topic"     

            assert result == expected_md_result
        def test_extract_returns_empty_string_on_empty_input(self, confluence_html_extractor, sample_url, sample_space_key):
            """
            Verifies that providing an empty HTML string results in an immediate early 
            return, bypassing the parsing logic entirely.
            
            This optimization ensures that BeautifulSoup is not initialized unnecessarily 
            for null inputs, improving performance and avoiding potential parsing errors 
            on empty data.
            """
            empty_html = ""

            with patch("knowledge_base.ingestion.confluence_content_extractor.BeautifulSoup") as mock_bs:
                result = confluence_html_extractor.extract(raw_html=empty_html, 
                                                           base_url = sample_url, 
                                                           space_key = sample_space_key)
                
            assert result == ""
            # Should not be called in this scenario
            mock_bs.assert_not_called()     
       
        def test_extract_removes_macro_metadata_noise(self, confluence_html_extractor, sample_url, sample_space_key):
            """
            Verifies that internal Confluence macro parameters (like 'BLOCK' metadata 
            and Table of Contents settings) are stripped from the output.
            
            This ensures that vector embeddings are not polluted with irrelevant 
            technical keywords like 'true', '20px', or 'atlassian-macro-output-type'.
            """
            
            raw_html = """
            <ac:structured-macro ac:name="excerpt">
            <ac:parameter ac:name="atlassian-macro-output-type">BLOCK</ac:parameter>
            <ac:rich-text-body><p>This is the real content.</p></ac:rich-text-body>
            </ac:structured-macro>
        
            <ac:structured-macro ac:name="toc">
            <ac:parameter ac:name="outline">true</ac:parameter>
            <ac:parameter ac:name="indent">20px</ac:parameter>
            </ac:structured-macro>"""

            result = confluence_html_extractor.extract(raw_html = raw_html, 
                                                       base_url = sample_url, 
                                                       space_key = sample_space_key)
            
            assert "This is the real content" in result
            
            assert "BLOCK" not in result    # Verify that metadata is removed
            assert "true" not in result     # Verify that TOC is removed
            assert "20px" not in result     # Verify that TOC is removed
        def test_extract_strips_all_images(self, confluence_html_extractor, sample_url, sample_space_key):
            """
            Verifies that <ac:image> tags and their attachments are completely removed, 
            leaving no visual or text artifacts in the markdown.
            
            This prevents image filenames (e.g., 'screenshot.jpg') or empty Markdown 
            image tags ('![]') from creating noise in the semantic search index.
            """
            raw_html = """
            <ac:image ac:alt="Useless Screenshot">
                <ri:attachment ri:filename="screenshot.jpg" />
            </ac:image>
            """
            result = confluence_html_extractor.extract(raw_html = raw_html, 
                                                       base_url = sample_url, 
                                                       space_key = sample_space_key)
            
            assert "screenshot.jpg" not in result   # Verify that image is removed
            assert "![" not in result               # Verify no MD image tags created

        def test_extract_unwraps_nested_layouts_cleanly(self, confluence_html_extractor, sample_url, sample_space_key):
            """
            Verifies that content buried inside complex Confluence layout tags (like 
            multi-column layouts) is successfully extracted and flattened into a single 
            readable stream.

            This ensures that text inside <ac:layout-cell> elements is not lost or 
            obscured by the XML structure during conversion, and that no XML layout 
            tags leak into the final output.
            """
            
            raw_html = """
                        <ac:layout>
                            <ac:layout-section ac:type="two_equal">
                                <ac:layout-cell>
                                    <h3>Left Column Header</h3>
                                    <p>This is content on the left.</p>
                                </ac:layout-cell>
                                <ac:layout-cell>
                                    <h3>Right Column Header</h3>
                                    <p>This is content on the right.</p>
                                </ac:layout-cell>
                            </ac:layout-section>
                        </ac:layout>
                        """
            result = confluence_html_extractor.extract(raw_html = raw_html, 
                                                       base_url = sample_url, 
                                                       space_key = sample_space_key)
            
            exp_fragment_1 = "### Left Column Header\n\nThis is content on the left."
            exp_fragment_2 = "### Right Column Header\n\nThis is content on the right."

            assert exp_fragment_1 in result
            assert exp_fragment_2 in result

            # Check against data leakage
            assert "ac:layout-cell" not in result
            assert "ac:layout" not in result

        def test_extract_removes_empty_paragraphs_and_whitespaces(self, confluence_html_extractor, sample_url, sample_space_key):
            """
            Verifies that the extractor aggressively cleans up "invisible" noise, such 
            as empty paragraphs, non-breaking spaces (&nbsp;), and pure whitespace blocks.
            
            This ensures that the final Markdown output is dense and semantically rich, 
            preventing the vector database from indexing empty strings or formatting 
            artifacts as meaningful content.
            """
            raw_html = """
            <p>&nbsp;</p>                 <p>   </p>                    <h1>   Real Content   </h1>   <p><br/></p>                  <div class="blank"></div>     """
            result = confluence_html_extractor.extract(raw_html = raw_html, 
                                                       base_url = sample_url, 
                                                       space_key = sample_space_key)

            expected = "# Real Content"

            assert result == expected
        def test_extract_handles_complex_links(self, confluence_html_extractor, sample_url, sample_space_key):
            """
            Verifies that Confluence 'magic' links (<ac:link>) are correctly converted 
            to absolute URLs, handling edge cases like empty link bodies, cross-space 
            references, and special characters in titles.

            Scenarios covered:
            1. Fallback to 'ri:content-title' when the link body is empty (CDATA Bug fix).
            2. Correct URL construction for links pointing to a different space.
            3. URL encoding of special characters (e.g., 'Q&A') in page titles.
            """
            raw_html = """
            <p>
                <ac:link>
                    <ri:page ri:content-title="Fallback Title" />
                    <ac:plain-text-link-body><![CDATA[]]></ac:plain-text-link-body>
                </ac:link>

                <ac:link>
                    <ri:page ri:content-title="Policies" ri:space-key="HR" />
                    <ac:plain-text-link-body>HR Policies</ac:plain-text-link-body>
                </ac:link>

                <ac:link>
                    <ri:page ri:content-title="Q&A: What is it?" />
                    <ac:plain-text-link-body>Click Here</ac:plain-text-link-body>
                </ac:link>
            </p>
            """
            result = confluence_html_extractor.extract(raw_html = raw_html, 
                                                       base_url = sample_url, 
                                                       space_key = sample_space_key)
            
            # Empty CDATA block
            assert "[Fallback Title]" in result

            # Non-default space key
            assert f"{sample_url}/display/HR/Policies" in result

            # Handles special characters
            # quote_plus uses ASCII URL Encoding - & becomes 26
            assert f"{sample_url}/display/{test}/Q%26A" in result




            

