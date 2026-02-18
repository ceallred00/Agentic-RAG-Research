import pytest
from unittest.mock import MagicMock, patch

from tools.perform_rag_tool import PerformRagTool, get_perform_rag_tool
from tools.rag_retriever import RagRetriever


class TestGetPerformRagTool:
    """Tests for the get_perform_rag_tool factory function."""

    @patch("tools.perform_rag_tool.PineconeSparseEmbedder")
    @patch("tools.perform_rag_tool.GeminiEmbedder")
    def test_factory_returns_configured_tool(self, MockGemini, MockSparse):
        """Factory injects a RagRetriever into the tool's _retriever attribute."""
        mock_execution_service = MagicMock()

        tool = get_perform_rag_tool(execution_service=mock_execution_service,
                                    index_name="test-index")

        assert isinstance(tool, PerformRagTool)
        assert isinstance(tool._retriever, RagRetriever)
        MockGemini.assert_called_once_with(mock_execution_service)
        MockSparse.assert_called_once_with(mock_execution_service)
        mock_execution_service.get_pinecone_client.assert_called_once()


class TestPerformRagToolRun:
    """Tests for PerformRagTool._run: delegation to RagRetriever and formatting."""

    def test_run_delegates_to_retriever_and_returns_formatted_string(self, mock_pinecone_matches):
        """_run calls retrieve_RAG_matches with correct args and returns formatted output."""
        # Arrange
        tool = PerformRagTool()
        mock_retriever = MagicMock()
        mock_retriever.retrieve_RAG_matches.return_value = mock_pinecone_matches
        tool._retriever = mock_retriever

        user_query = "What is the deadline for dropping a class?"
        top_k = 3

        # Act
        result = tool._run(user_query=user_query, top_k_matches=top_k)

        # Assert
        mock_retriever.retrieve_RAG_matches.assert_called_once_with(
            user_query=user_query,
            top_k_matches=top_k
        )
        # Result should be a formatted string containing match content
        assert "--- Result" in result
        assert "0.89" in result


class TestFormatResults:
    """Tests for PerformRagTool._format_results: match formatting, missing metadata, and edge cases."""

    @pytest.fixture
    def tool(self):
        """PerformRagTool instance (no retriever needed for format tests)."""
        return PerformRagTool()

    def test_formats_multiple_matches(self, tool, mock_pinecone_matches):
        """Multiple matches are formatted with scores and text content."""
        result = tool._format_results(mock_pinecone_matches)

        assert "--- Result (Score: 0.89) ---" in result
        assert "--- Result (Score: 0.75) ---" in result
        assert "The last day to drop without a W is August 25th." in result
        assert "Drop/Add period ends on the first week." in result

    def test_missing_text_metadata(self, tool):
        """Match with no 'text' key in metadata falls back to default message."""
        match = MagicMock()
        match.metadata = {"source": "some_doc.pdf"}
        match.score = 0.50

        result = tool._format_results([match])

        assert "No content available." in result

    def test_no_score_shows_na(self, tool):
        """Match with score=None displays 'N/A'."""
        match = MagicMock()
        match.metadata = {"text": "Some content here."}
        match.score = None

        result = tool._format_results([match])

        assert "--- Result (Score: N/A) ---" in result
        assert "Some content here." in result

    def test_zero_score_shows_na(self, tool):
        """Match with score=0 displays 'N/A' (falsy value)."""
        match = MagicMock()
        match.metadata = {"text": "Some content here."}
        match.score = 0

        result = tool._format_results([match])

        assert "--- Result (Score: N/A) ---" in result

    def test_empty_matches_returns_empty_string(self, tool):
        """Empty matches list returns an empty string."""
        result = tool._format_results([])

        assert result == ""

    def test_none_metadata_uses_empty_dict(self, tool):
        """Match with metadata=None does not raise; falls back to default message."""
        match = MagicMock()
        match.metadata = None
        match.score = 0.60

        result = tool._format_results([match])

        assert "No content available." in result
