import requests
import logging
import time
from typing import Set, List, Union, Optional, Dict
from knowledge_base.ingestion.confluence_page_processor import ConfluencePageProcessor
from constants import UWF_PUBLIC_KB_PROCESSED_DATE_DIR, UWF_CONFLUENCE_BASE_URL
from pathlib import Path

# For main entry point
from core.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class URLtoMarkdownConverter:
    """
    Recursively scrapes a Confluence page tree via JSON API.

    1. Recursively traverse page tree.
    2. Fetch JSON data.
    3. Delegate processing to the PageProcessor
    4. Save markdown files.
    """

    def __init__(
        self,
        base_url: str = UWF_CONFLUENCE_BASE_URL,
        saved_data_path: Union[str, Path] = UWF_PUBLIC_KB_PROCESSED_DATE_DIR,
    ):
        """
        Initializes the converter with API session settings and processing tools.

        Args:
            base_url (str): The root URL of the Confluence instance.
            saved_data_path (Union[str, Path]): Directory where markdown files will be saved.
        """
        self.base_url = base_url
        self.processor = ConfluencePageProcessor(saved_data_path=saved_data_path)

        # body.storage returns the raw HTML from the Confluence API - not including website HTML
        # body.version returns the version number and information about the last modification made to the page.
        self.params = {"expand": "body.storage,version"}

        self.visited_ids: Set[str] = set()

        self.session = requests.Session()  # Setup persistent session
        self.session.headers.update({"Accept": "application/json"})

    def scrape_tree(self, root_id: Union[int, str]):
        """
        The entry point for the scraping process. Fetches the root page and initiates recursion.

        Actions:
        1. Normalizes the `root_id` to a string.
        2. Fetches the metadata for the root page itself.
        3. Processes the root page (saves it as Markdown).
        4. Initializes the ancestor chain (starting empty for the root).
        5. Calls `recursively_crawl_tree` to handle all descendants.

        Args:
            root_id (Union[int, str]): The ID of the page where scraping begins.

        Returns:
            None
        """
        root_id = str(root_id)
        url = f"{self.base_url}/rest/api/content/{root_id}"
        logger.info(f"Beginning to scrape: {url}")

        root_data = self._api_request(url, root_id)

        if root_data:
            title = root_data.get("title", "ROOT")

            # Root page has no ancestors
            initial_ancestors = []

            self.processor.process_page(
                child_data=root_data,
                ancestors=initial_ancestors,
                base_url=self.base_url,
            )

            logger.info(f"Root '{title}' processed. Beginning recursive crawl...")
            # Add Root to the ancestors list
            current_ancestors = [{"id": root_id, "title": title}]

            self.recursively_crawl_tree(parent_id=root_id, ancestors=current_ancestors)
        else:
            logger.critical(f"Could not fetch root page {root_id}. Aborting crawl.")

    def recursively_crawl_tree(self, parent_id: str, ancestors: List[Dict[str, str]]):
        """
        Recursively traverses the page tree depth-first.

        Actions:
        1. Checks if `parent_id` has already been visited to prevent infinite loops.
        2. Fetches all immediate children of the parent using `_fetch_immediate_children`.
        3. Iterates through every child found.
        4. Delegates the processing of the child to `self.processor`.
        5. Updates the `ancestors` list by appending the current child.
        6. Recurses into the child (child becomes the new parent).

        Args:
            parent_id (str): The ID of the page whose children we are fetching.
            ancestors (List[Dict[str, str]]): A list of dictionaries representing the breadcrumb trail.
                Example: [{'id': '10', 'title': 'Root'}, {'id': '20', 'title': 'Child'}]

        Returns:
            None
        """
        # Prevents circular references A -> B -> A
        parent_id = str(parent_id)  # Type enforcement - For Set
        if parent_id in self.visited_ids:
            return
        self.visited_ids.add(parent_id)

        logger.info(f"Fetching children for {parent_id}")
        children = self._fetch_immediate_children(parent_id)

        if not children:
            logger.debug(f"No children found for ID {parent_id}. Skipping.")

        for child in children:
            # Protects against API returning None value
            page_id = str(child.get("id") or "")
            page_title = str(child.get("title") or "")

            self.processor.process_page(child_data=child, ancestors=ancestors, base_url=self.base_url)

            next_level_ancestors = ancestors + [{"id": page_id, "title": page_title}]

            time.sleep(0.2)
            self.recursively_crawl_tree(parent_id=str(page_id), ancestors=next_level_ancestors)

    def _fetch_immediate_children(self, parent_id: str) -> List[dict]:
        """
        Fetches all immediate child pages of a specific parent, handling API pagination.

        Actions:
        1. Constructs the initial API URL for child pages.
        2. Enters a `while` loop to handle pagination (Confluence limits results per request - Default 25).
        3. Fetches data and appends results to `all_pages`.
        4. Checks the response for a `_links.next` URL.
        5. If a next link exists, updates `api_url` and repeats; otherwise, breaks.

        Args:
            parent_id (str): The numerical ID of the parent page.

        Returns:
            List[dict]: A list of raw page objects from the Confluence API (Docs: https://docs.atlassian.com/atlassian-confluence/REST/6.6.0/#content-getContentById)

            Example Return Object:
            [
                {
                    "id": "12345",
                    "title": "Advising Syllabus",
                    "type": "page",
                    "status": "current",
                    "body": {
                        "storage": { "value": "<p>Raw HTML content...</p>" }
                    },
                    "version": {
                        "number": 2,
                        "when": "2023-11-27T12:05:17.897-06:00"
                    }
                },
                ...
            ]
        """
        api_url = f"{self.base_url}/rest/api/content/{parent_id}/child/page"

        all_pages = []

        while api_url:
            logger.info(f"Fetching children for URL: {api_url}")

            data = self._api_request(api_url, parent_id)

            if not data:
                break  # Break loop if request fails

            results = data.get("results", [])  # Returns list[dict]

            all_pages.extend(results)

            # Handles pagination
            next_link = data.get("_links", {}).get("next")

            if next_link:
                # Confluence returns a relative path like '/rest/api/content/...'
                # Join it with the base url to get the full URL for the next loop
                api_url = f"{self.base_url}{next_link}"
            else:
                # No more pages left!
                api_url = None

        return all_pages

    def _api_request(self, url: str, parent_id) -> Optional[Dict]:
        """
        Executes a safe HTTP GET request with error handling.

        Actions:
        1. Sends a GET request to the provided URL.
        2. Raises an exception for HTTP error codes (4xx, 5xx).
        3. Catches specific exceptions (Timeout, ConnectionError, JSONDecodeError).
        4. Logs errors with the associated `parent_id` for context.

        Args:
            url (str): The full URL to fetch.
            parent_id (Union[str, int]): Used only for logging context.

        Returns:
            Optional[Dict]: The parsed JSON response as a dictionary, or None if failed.
        """
        try:
            response = self.session.get(url, params=self.params, timeout=30)  # 30 second timeout
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            # Catches 401 (Auth), 403 (Forbidden), 404 (Not Found)
            logger.error(f"HTTP error occurred for parent id {parent_id}: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            # Catches DNS issues or refused connections
            logger.error(f"Connection error occurred for parent id {parent_id}: {conn_err}")
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for parent id {parent_id}")
        except requests.exceptions.JSONDecodeError as json_err:
            logger.error(f"JSON decoding error occurred for parent id {parent_id}: {json_err}")
        except requests.exceptions.RequestException as err:
            logger.error(f"An unexpected request error occurred for id {parent_id}: {err}")

        return None


if __name__ == "__main__":  # pragma: no cover
    # setup_logging()

    converter = URLtoMarkdownConverter(
        base_url=UWF_CONFLUENCE_BASE_URL,
        saved_data_path=UWF_PUBLIC_KB_PROCESSED_DATE_DIR,
    )

    converter.scrape_tree(7641671)
