import logging
from knowledge_base.ingestion.confluence_content_extractor import ConfluenceContentExtractor
from knowledge_base.ingestion.file_saver import FileSaver
from pathlib import Path
from typing import Union, List, Dict
from constants import UWF_PUBLIC_KB_PROCESSED_DATE_DIR, UWF_CONFLUENCE_PAGE_SPACE, UWF_CONFLUENCE_BASE_URL

logger = logging.getLogger(__name__)

class ConfluencePageProcessor:
    """
    Handles transformation pipeline:

    Raw Data -> Clean Markdown -> Metadata Injection -> File System
    """
    def __init__(self, saved_data_path: Union[str, Path] = UWF_PUBLIC_KB_PROCESSED_DATE_DIR):
        self.file_saver = FileSaver(saved_data_path)
        self.content_extractor = ConfluenceContentExtractor()
    
    def process_page(self, child_data: dict, ancestors: List[Dict[str, str]], base_url: str = UWF_CONFLUENCE_BASE_URL):
        """
        Processes a single page object, converts it to Markdown, and saves it.

        Actions:
        1. Extracts extracted metadata (ID, Title, Version, Last Updated) from `child_data`.
        2. Validates that the page has content; if not, logs a warning and returns early.
        3. Passes the raw HTML to `ConfluenceContentExtractor` to get clean Markdown.
        4. Constructs the breadcrumb path string from the `ancestors` list.
        5. Formats the final content string with YAML Frontmatter.
        6. Delegates saving to `FileSaver`.

        Args:
            child_data (dict): The raw dictionary for a single page returned by Confluence API.
            ancestors (List[Dict[str, str]]): A list of dictionaries containing title/id of parent pages.
                Example: [{'id': '1', 'title': 'Root'}, {'id': '2', 'title': 'Sub-Page'}]
            base_url (str): The base URL (e.g., https://confluence.uwf.edu) used to construct the permalink.

        Returns:
            None: This function performs side-effects (IO) and does not return a value.
        """
        page_id = str(child_data.get('id') or "")
        page_title = str(child_data.get('title') or "")

        # Defaults
        version_num = 1
        last_updated = "" 

        version_data = child_data.get('version', {})
        if version_data:
            version_num = version_data.get('number', version_num)
            last_updated = version_data.get('when',last_updated)
        
        # Used for internal link navigation when parsing HTML (ConfluenceContentExtractor)
        # Defaults to UWF_CONFLUENCE_PAGE_SPACE if not found
        current_space = child_data.get('space', {}).get('key', UWF_CONFLUENCE_PAGE_SPACE)
        
        raw_html = child_data.get('body', {}).get('storage', {}).get('value',"")

        if not raw_html:
            logger.warning(f"Skipping '{page_title}' (ID: {page_id}) - No content found.")
            return
        
        cleaned_markdown = self.content_extractor.extract(raw_html, base_url = base_url, space_key = current_space)

        path_string = " / ".join([a['title'] for a in ancestors])

        immediate_parent = ancestors[-1]['title'] if ancestors else "None"

        full_url = f"{base_url}/pages/viewpage.action?pageId={page_id}"

        # Metadata added as YAML Frontmatter
        final_content = (
                    f"---\n"
                    f"title: {page_title}\n"
                    f"parent: {immediate_parent}\n"
                    f"path: {path_string}\n"
                    f"original_url: {full_url}\n"
                    f"page_id: {page_id}\n"
                    f"version: {version_num}\n"
                    f"last_updated: {last_updated}\n"
                    f"---\n\n"
                    f"{cleaned_markdown}"
                )

        self.file_saver.save_markdown_file(final_content,page_title)