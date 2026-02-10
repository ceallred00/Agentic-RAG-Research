import pytest
import logging
from unittest.mock import MagicMock

@pytest.fixture
def mock_cleaned_html():
    return "Raw HTML content..."


class TestConfluencePageProcessor:
    class TestProcessPage:
        def _expected_values(self, confluence_page, ancestors, cleaned_html, url):
            """
            Helper to construct the expected file content string and title for assertions.

            This mirrors the logic inside `ConfluencePageProcessor` to generate the 
            YAML frontmatter and full Markdown content based on the input data.

            Args:
                confluence_page (dict): The dictionary representation of the Confluence page.
                ancestors (list): A list of ancestor dictionaries.
                cleaned_html (str): The mock HTML content string.
                url (str): The base URL used for the 'original_url' field.

            Returns:
                tuple: A tuple containing:
                    - expected_content (str): The full expected Markdown string.
                    - expected_page_title (str): The expected title of the page.
            """
            expected_page_id = str(confluence_page.get('id') or "")
            expected_page_title = str(confluence_page.get('title') or "")

            # In line with expected default values 
            expected_version_num = confluence_page.get('version', {}).get('number', 1)
            expected_last_updated = confluence_page.get('version', {}).get('when', "")

            expected_parent = ancestors[-1]['title'] if ancestors else "None"
            expected_path_string = " / ".join([a['title'] for a in ancestors] + [expected_page_title])            
            
            expected_content = (
                "---\n"
                f"title: {expected_page_title}\n"
                f"parent: {expected_parent}\n"
                f"path: {expected_path_string}\n"
                f"original_url: {url}/pages/viewpage.action?pageId={expected_page_id}\n"
                f"page_id: {expected_page_id}\n"
                f"version: {expected_version_num}\n"
                f"last_updated: {expected_last_updated}\n"
                "---\n\n"
                f"{cleaned_html}"
            )

            return expected_content, expected_page_title

        def test_process_page_happy_path(self, sample_confluence_page_json, confluence_processor, sample_url, sample_ancestors, mock_cleaned_html):
            """
            Verifies the end-to-end "happy path" for processing a single Confluence page.
            
            This test ensures that when valid page data and ancestors are provided:
            1. Metadata (title, ID, version, last updated) is correctly extracted.
            2. The content extractor is invoked to convert HTML to Markdown.
            3. YAML frontmatter is dynamically constructed using the correct ancestor path.
            4. The file saver is called exactly once with the properly formatted file content.
            """            
            confluence_processor.file_saver.save_markdown_file = MagicMock()
            
            confluence_processor.content_extractor.extract.return_value = mock_cleaned_html

            result = confluence_processor.process_page(
                child_data = sample_confluence_page_json,
                ancestors = sample_ancestors, 
                base_url = sample_url
            )

            expected_content, expected_page_title = self._expected_values(confluence_page=sample_confluence_page_json, 
                                                     ancestors = sample_ancestors,
                                                     cleaned_html = mock_cleaned_html,
                                                     url = sample_url)
            

            confluence_processor.file_saver.save_markdown_file.assert_called_once_with(expected_content, expected_page_title)
            assert result is None

        def test_process_page_skips_empty_content(self, sample_confluence_page_json, confluence_processor, caplog, sample_url, sample_ancestors):
            """
            Verifies that pages with empty or missing content bodies are skipped gracefully.
            
            This test ensures that:
            1. The method logs a specific warning identifying the skipped page.
            2. The execution returns early (returning None).
            3. Costly downstream operations (like the content extractor) are NOT called.
            """
            sample_confluence_page_json["body"]["storage"] = {}
            
            with caplog.at_level(logging.WARNING):
                results = confluence_processor.process_page(
                    child_data = sample_confluence_page_json,
                    ancestors = sample_ancestors,
                    base_url = sample_url
                )
            # Should not reach this - Early exit
            confluence_processor.content_extractor.extract.assert_not_called()
            assert results is None
            assert "No content found" in caplog.text
        def test_process_page_missing_metadata(self, confluence_processor, sample_confluence_page_json, sample_url, sample_ancestors, mock_cleaned_html):
            """
            Verifies that the processor applies safe defaults when optional metadata is missing.

            This test removes the 'version' key and empties the 'id' and 'title' fields
            to ensure the code:
            1. Defaults version number to 1 and last_updated to an empty string.
            2. Handles empty ID/Title strings without crashing.
            3. Successfully calls the file saver even with minimal metadata.
            """
            sample_confluence_page_json["id"] = ""
            sample_confluence_page_json["title"] = ""
            del sample_confluence_page_json["version"]

            confluence_processor.content_extractor.extract.return_value = mock_cleaned_html
            # Sending an empty title string to the save_markdown_file function should cause an early return (None)
            confluence_processor.file_saver.save_markdown_file = MagicMock()
            
            confluence_processor.process_page(child_data = sample_confluence_page_json, 
                                              ancestors = sample_ancestors,
                                              base_url = sample_url)
            
            expected_content, expected_page_title = self._expected_values(confluence_page=sample_confluence_page_json, 
                                                     ancestors = sample_ancestors,
                                                     cleaned_html = mock_cleaned_html,
                                                     url = sample_url)
            
            
            assert expected_page_title == ""
            confluence_processor.file_saver.save_markdown_file.assert_called_once_with(expected_content, expected_page_title)
        @pytest.mark.parametrize("ancestors, case_description",[
            ([], "Root Page (No Ancestors)"),
            ([{"id": "12345", "title": "Grandparent"},{"id": "6789", "title": "Parent"}], "Nested Page (Multiple Ancestors)")
        ])
        def test_process_page_ancestor_handling(self, confluence_processor, sample_url, mock_cleaned_html, sample_confluence_page_json, ancestors, case_description):
            """
            Verifies ancestor processing for both Root pages and Nested pages.
            
            Uses parameterization to test:
            1. Root Page: ancestors=[], expects parent="None"
            2. Nested Page: ancestors=[...], expects parent="Parent" and correct path string

            case_description passed as an argument for debugging purposes in case of failed tests
            """
            confluence_processor.content_extractor.extract.return_value = mock_cleaned_html
            confluence_processor.file_saver.save_markdown_file = MagicMock()

            confluence_processor.process_page(child_data = sample_confluence_page_json, 
                                              ancestors = ancestors,
                                              base_url = sample_url)
            
            expected_content, expected_page_title = self._expected_values(confluence_page= sample_confluence_page_json, 
                                                                          ancestors = ancestors, 
                                                                          cleaned_html = mock_cleaned_html,
                                                                          url = sample_url)
            
            confluence_processor.file_saver.save_markdown_file.assert_called_once_with(expected_content, expected_page_title)
        def test_process_page_cleans_artifacts(self, confluence_processor, sample_confluence_page_json, sample_url, sample_ancestors):
            """
            Verifies that the processor creates a clean file even when the extractor returns dirty text.

            Tests against any left over XML/HTML artifacts
            """
            # The extractor returns "Dirty" text
            dirty_input = "Click here [>]]>](http://bad) for more info.]]>"
            confluence_processor.content_extractor.extract.return_value = dirty_input
            
            confluence_processor.file_saver.save_markdown_file = MagicMock()

            confluence_processor.process_page(
                child_data=sample_confluence_page_json,
                ancestors=sample_ancestors,
                base_url=sample_url
            )

            expected_clean_text = "Click here  for more info." 

            expected_content, _ = self._expected_values(
                confluence_page=sample_confluence_page_json, 
                ancestors=sample_ancestors,
                cleaned_html=expected_clean_text,
                url=sample_url
            )

            confluence_processor.file_saver.save_markdown_file.assert_called_once_with(expected_content, "Advising Syllabus")
