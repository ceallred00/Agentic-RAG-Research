import pytest
import logging
import requests
from unittest.mock import MagicMock, patch
from typing import Set

class TestURLToMDConverter:
    class TestApiRequest:
        def test_api_request_happy_path(self, url_converter, sample_url, sample_confluence_page_json, sample_parent_id):
            """
            Verifies that a successful API request returns the expected JSON data.

            Ensures that when the server returns a 200 OK status:
            1. The underlying `session.get` is called with the correct URL, parameters, and timeout.
            2. The method returns the parsed JSON dictionary from the response.
            """
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_confluence_page_json

            url_converter.session.get.return_value = mock_response

            response = url_converter._api_request(url = sample_url, parent_id = sample_parent_id)

            assert response == sample_confluence_page_json
            url_converter.session.get.assert_called_once_with(sample_url, params = url_converter.params, timeout = 30)
        
        @pytest.mark.parametrize("exception_class, log_snippet, error_source", [
            # Error is raised by .raise_for_status() method
            (requests.exceptions.HTTPError("404 Client Error"), "HTTP error", "status"),
            # Error is raised by session.get
            (requests.exceptions.ConnectionError("Connection Refused"), "Connection error", "session"),
            (requests.exceptions.Timeout("Read Timeout"), "Request timed out", "session"),
            (requests.exceptions.RequestException("Unknown Error"), "unexpected request error", "session"),

            # Error is raised by response.json()
            (requests.exceptions.JSONDecodeError("Msg", "bad JSON", 0), "JSON decoding error", "json")
        ])        
        def test_api_request_network_errors(self, url_converter, sample_url, caplog, exception_class, log_snippet, error_source, sample_parent_id):
            """
            Verifies robust error handling for Network, HTTP, and Data parsing failures.

            Using parameterization, this test ensures that:
            1. HTTP Errors (e.g., 404) raised by `raise_for_status` are caught.
            2. Network Errors (Connection, Timeout) raised by `session.get` are caught.
            3. Malformed Data Errors (JSONDecodeError) are caught.
            4. In all cases, the specific error is logged (with the parent ID) and `None` is returned to prevent a crash.
            """
            # Reset between parameterized tests
            url_converter.session.get.reset_mock()

            mock_response = MagicMock()
            mock_response.status_code = 200

            if error_source == "status":
                # For HTTP errors, raise_for_status() fails
                mock_response.status_code = 404
                mock_response.raise_for_status.side_effect = exception_class
                url_converter.session.get.return_value = mock_response

            elif error_source == "session":
                # For network errrors, session.get() fails
                url_converter.session.get.side_effect = exception_class
            
            elif error_source == "json":
                # For JSON errors, response.json() fails
                mock_response.json.side_effect = exception_class
                url_converter.session.get.return_value = mock_response
            
            with caplog.at_level(logging.ERROR):
                result = url_converter._api_request(sample_url, parent_id = sample_parent_id)
            
            assert result is None
            assert log_snippet in caplog.text
            assert sample_parent_id in caplog.text
    class TestFetchImmediateChildren:
        def test_fetch__children_happy_path_single_page(self, url_converter, sample_confluence_page_json, sample_parent_id):
            """
            Verifies retrieving children when all results fit on a single API response page.

            Ensures that:
            1. The method accepts the standard Confluence API response format.
            2. It correctly extracts the list of children from the 'results' key.
            3. Only one API call is made when the '_links' dictionary is empty.
            """
            api_response = {"results": [sample_confluence_page_json], "_links": {}}

            url_converter._api_request = MagicMock(return_value = api_response)

            children = url_converter._fetch_immediate_children(sample_parent_id)

            assert len(children) == 1
            assert children[0] == sample_confluence_page_json
            url_converter._api_request.assert_called_once()
        def test_fetch_children_pagination(self, url_converter, sample_confluence_page_json, sample_parent_id):
            """
            Verifies that the method follows pagination links to aggregate all children.

            Simulates a multi-page scenario (Page 1 -> Page 2 -> End) and ensures:
            1. The method detects the 'next' link in the response metadata.
            2. It constructs the correct absolute URL for the subsequent request.
            3. It loops until no 'next' link remains.
            4. The results from all pages are combined into a single flat list.
            """
            next_link = "/rest/api/content/next_page"
            page_1_response = {"results": [sample_confluence_page_json], "_links": {"next": next_link}}
            page_2_response = {"results": [sample_confluence_page_json], "_links": {}}

            url_converter._api_request = MagicMock(side_effect=[page_1_response, page_2_response])

            children = url_converter._fetch_immediate_children(sample_parent_id)

            assert len(children) == 2 # 2 total results - One from page 1 and one from page 2
            assert url_converter._api_request.call_count == 2
            # Verify page content
            assert children[0] == sample_confluence_page_json
            assert children[1] == sample_confluence_page_json

            # Verify the second method call was called with the correct URL 
            expected_next_url = f"{url_converter.base_url}{next_link}"

            second_call = url_converter._api_request.call_args_list[1]
            # .args returns positional arguments
            assert second_call.args[0] == expected_next_url
        def test_fetch_children_no_data(self, url_converter, sample_parent_id):
            """
            Verifies the base case where a parent page has no children.

            Ensures that when the API returns an empty 'results' list:
            1. The method returns an empty list (not None).
            2. No errors are raised during processing.
            """
            empty_response = {"results": [], "_links": {}}
            url_converter._api_request = MagicMock(return_value = empty_response)

            children = url_converter._fetch_immediate_children(sample_parent_id)

            assert children == []
            assert len(children) == 0
            url_converter._api_request.assert_called_once()
    class TestRecursivelyCrawlTree:
        @pytest.fixture(autouse=True)
        def mock_sleep(self):
            """Automatically mocks time.sleep for all tests in this class to speed up execution."""
            with patch('time.sleep'):
                yield
        def test_recursive_crawl_happy_path(self, url_converter, sample_ancestors, sample_confluence_page_json, sample_parent_id):
            """
            Verifies the standard recursive flow where a parent node has children.

            This test simulates a tree with depth 1 (Parent -> Child -> [No Children]) and asserts:
            1. The processor is called exactly once for the child page with correct arguments.
            2. The crawler recurses, calling `_fetch_immediate_children` for both the parent and the child.
            3. Both the parent and child IDs are correctly tracked in `visited_ids`.
            """
            child_page = sample_confluence_page_json
            child_id = child_page['id']
            # First call returns a dictionary of children, the second call returns no children (stops recursion)
            url_converter._fetch_immediate_children = MagicMock(side_effect = [[child_page],[]])
            # This function extracts the HTML, converts into MD, and saves - No need for a sample response here.
            url_converter.processor.process_page = MagicMock()

            url_converter.recursively_crawl_tree(parent_id = sample_parent_id, ancestors = sample_ancestors)

            # Should only be called once during first function call
            assert url_converter.processor.process_page.call_count == 1

            # Grab the first (only) call
            first_process_page_call = url_converter.processor.process_page.call_args_list[0]
            # The call object returns (pos_args(tuple), kwargs(dict))
            assert first_process_page_call.kwargs['child_data'] == sample_confluence_page_json
            assert first_process_page_call.kwargs['ancestors'] == sample_ancestors
            assert first_process_page_call.kwargs['base_url'] == url_converter.base_url

            assert url_converter._fetch_immediate_children.call_count == 2

            # Verify that _fetch_immediate_children was called twice with the correct arguments
            # .args grabs the positional arguments
            assert url_converter._fetch_immediate_children.call_args_list[0].args[0] == sample_parent_id
            # Should be parsed from the child returned by the first function call.
            assert url_converter._fetch_immediate_children.call_args_list[1].args[0] == child_id

            assert sample_parent_id in url_converter.visited_ids
            assert child_id in url_converter.visited_ids
            
        def test_recursive_crawl_circular(self, url_converter, sample_parent_id, sample_ancestors):
            """
            Verifies that the crawler detects and breaks circular references.

            Ensures that if a `parent_id` has already been marked in `visited_ids`:
            1. The function returns immediately without fetching children or processing.
            2. Infinite recursion loops are prevented.
            3. No network calls are made (verified by the side_effect Exception).
            """
            url_converter.visited_ids = {sample_parent_id, "12345"}

            url_converter._fetch_immediate_children = MagicMock(side_effect=Exception("Should not reach this line"))

            response = url_converter.recursively_crawl_tree(parent_id = sample_parent_id, ancestors = sample_ancestors)

            assert response is None
            assert sample_parent_id in url_converter.visited_ids
        def test_crawl_leaf_with_no_children(self, caplog, url_converter, sample_parent_id, sample_ancestors):
            """
            Verifies the base case where a page has no children (a leaf node).

            Ensures that when `_fetch_immediate_children` returns an empty list:
            1. The recursion stops immediately.
            2. No processing attempts are made (process_page is not called).
            3. The current node is still marked as visited to prevent future cycles.
            """
            url_converter._fetch_immediate_children = MagicMock(return_value = [])
            url_converter.processor.process_page = MagicMock()

            with caplog.at_level(logging.DEBUG):
                result = url_converter.recursively_crawl_tree(parent_id = sample_parent_id, ancestors = sample_ancestors)
            
            assert result is None
            # Should be called once 
            url_converter._fetch_immediate_children.assert_called_once()
            # Should not reach processor.process_page()
            url_converter.processor.process_page.assert_not_called()
            assert sample_parent_id in url_converter.visited_ids
        def test_recursive_crawl_handles_missing_child_fields(self, url_converter, sample_ancestors, sample_parent_id):
            """
            Verifies robustness against malformed child data (missing ID or Title).

            Ensures that when the API returns a child object missing standard keys:
            1. The code does not crash (KeyError or AttributeError).
            2. `None` values are sanitized into empty strings.
            3. The empty string is correctly passed to the recursive call, ensuring the crawler attempts to continue safely.
            """
            malformed_child = {"type": "page"} # No 'id' or 'title'
            
            # First call returns a dictionary of children, the second call returns no children (stops recursion)
            url_converter._fetch_immediate_children = MagicMock(side_effect=[[malformed_child], []])
            url_converter.processor.process_page = MagicMock()

            url_converter.recursively_crawl_tree(sample_parent_id, sample_ancestors)

            # It should still process the page (even with empty strings)
            url_converter.processor.process_page.assert_called_once()
            
            # Verify it passed the child dict as is (no handling of missing fields)
            call_args = url_converter.processor.process_page.call_args[1]
            assert call_args['child_data'] == malformed_child
            
            # Verify that the code recursed with empty strings
            assert url_converter._fetch_immediate_children.call_count == 2
            # Verif that the second function call passed an empty string for the parent_id.
            children_call_args = url_converter._fetch_immediate_children.call_args_list[1]
            # Grabs positional arguments
            assert children_call_args.args[0] == ""
    class TestScrapeTree:
        def test_scrape_tree_happy_path(self, url_converter, sample_confluence_page_json, sample_ancestors):
            """
            Verifies the successful bootstrapping of the crawl from the root node.

            Ensures that when valid root data is returned:
            1. The root page itself is processed immediately (with an empty ancestor list).
            2. The root page is added to the ancestor history.
            3. The recursion is kicked off by calling `recursively_crawl_tree` with the root as the parent.
            """
            url_converter._api_request = MagicMock(return_value = sample_confluence_page_json)
            url_converter.processor.process_page = MagicMock()
            url_converter.recursively_crawl_tree = MagicMock()

            root_id = str(sample_confluence_page_json.get("id"))
            root_title = sample_confluence_page_json.get("title")

            url_converter.scrape_tree(root_id = root_id)

            assert url_converter._api_request.call_count == 1

            assert url_converter.processor.process_page.call_count == 1
            
            # Grab the first (only) call
            call = url_converter.processor.process_page.call_args_list[0]
            assert call.kwargs['child_data'] == sample_confluence_page_json
            assert call.kwargs['ancestors'] == []
            assert call.kwargs['base_url'] == url_converter.base_url

            assert url_converter.recursively_crawl_tree.call_count == 1
            # Grab first (only) call
            crawl_call = url_converter.recursively_crawl_tree.call_args_list[0]
            assert crawl_call.kwargs['parent_id'] == root_id
            assert crawl_call.kwargs['ancestors'] == [{"id":root_id, "title": root_title}]

        def test_scrape_tree_no_data(self, url_converter, caplog, sample_parent_id):
            """
            Verifies that the process aborts safely if the root page cannot be fetched.

            Ensures that when the API returns `None` for the root ID:
            1. A critical error is logged containing the missing ID.
            2. The processing and recursive crawling steps are skipped entirely.
            3. The program exits the function without crashing.
            """
            url_converter._api_request = MagicMock(return_value = None)
            url_converter.recursively_crawl_tree = MagicMock()
            with caplog.at_level(logging.CRITICAL):
                url_converter.scrape_tree(sample_parent_id)

            assert f"Could not fetch root page {sample_parent_id}" in caplog.text
            
            url_converter._api_request.assert_called_once()
            # Should not make it to this point.
            assert url_converter.recursively_crawl_tree.call_count == 0
            



