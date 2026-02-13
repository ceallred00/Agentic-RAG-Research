import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from pinecone.exceptions import PineconeApiException
from knowledge_base.processing.pinecone_sparse_embedder import PineconeSparseEmbedder


class TestPineconeSparseEmbedderRetry:
    """Tests for rate limit retry behavior in _create_embeddings."""

    @pytest.fixture
    def embedder_with_mock_client(self, instance_execution_service, mock_pinecone_client):
        """Creates a PineconeSparseEmbedder with a mock Pinecone client for testing."""
        mock_client_instance = mock_pinecone_client.return_value
        embedder = PineconeSparseEmbedder(instance_execution_service)
        embedder.pinecone_client = mock_client_instance
        return embedder, mock_client_instance

    @patch("knowledge_base.processing.pinecone_sparse_embedder.time.sleep")
    @patch("knowledge_base.processing.pinecone_sparse_embedder.retry_with_backoff")
    def test_429_triggers_retry(self, mock_retry, mock_sleep, embedder_with_mock_client):
        """Verifies that a 429 PineconeApiException triggers retry_with_backoff instead of raising."""
        embedder, mock_client = embedder_with_mock_client

        rate_limit_error = PineconeApiException(status=429, reason="Too Many Requests")
        mock_client.inference.embed.side_effect = rate_limit_error

        mock_sparse = MagicMock()
        mock_sparse.sparse_indices = [1, 2]
        mock_sparse.sparse_values = [0.5, 0.3]
        mock_retry.return_value = [mock_sparse]

        embedder._create_embeddings(task_type="passage", inputs=[Document(page_content="test")])

        mock_retry.assert_called_once()
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["retryable_on"] == (PineconeApiException,)
        assert call_kwargs.kwargs["max_retries"] == 6

    @patch("knowledge_base.processing.pinecone_sparse_embedder.time.sleep")
    def test_non_429_pinecone_error_raises(self, mock_sleep, embedder_with_mock_client):
        """Verifies that a non-429 PineconeApiException (e.g., 400) propagates immediately without retry."""
        embedder, mock_client = embedder_with_mock_client

        bad_request_error = PineconeApiException(status=400, reason="Bad Request")
        mock_client.inference.embed.side_effect = bad_request_error

        with pytest.raises(PineconeApiException):
            embedder._create_embeddings(task_type="passage", inputs=[Document(page_content="test")])

    @patch("knowledge_base.processing.pinecone_sparse_embedder.time.sleep")
    def test_generic_exception_raises_runtime_error(self, mock_sleep, embedder_with_mock_client):
        """Verifies that a non-PineconeApiException is wrapped in RuntimeError."""
        embedder, mock_client = embedder_with_mock_client

        mock_client.inference.embed.side_effect = ConnectionError("network failure")

        with pytest.raises(RuntimeError, match="network failure"):
            embedder._create_embeddings(task_type="passage", inputs=[Document(page_content="test")])

    @patch("knowledge_base.processing.pinecone_sparse_embedder.time.sleep")
    def test_successful_batch_logs_and_throttles(self, mock_sleep, embedder_with_mock_client):
        """Verifies that successful batches are followed by a 0.5s throttle sleep."""
        embedder, mock_client = embedder_with_mock_client

        mock_sparse = MagicMock()
        mock_sparse.sparse_indices = [1]
        mock_sparse.sparse_values = [0.5]
        mock_client.inference.embed.return_value = [mock_sparse]

        embedder._create_embeddings(task_type="passage", inputs=[Document(page_content="test")])

        # Verify throttle sleep was called
        mock_sleep.assert_called_with(0.5)
