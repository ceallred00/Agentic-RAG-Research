import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_google_genai._common import GoogleGenerativeAIError
from knowledge_base.processing.gemini_embedder import GeminiEmbedder


class TestGeminiEmbedder:
    def test_doc_client_configuration(self, instance_execution_service):
        pass

    def test_query_client_configuration(self, instance_execution_service):
        pass

    def test_embed_KB_document_dense_batching(
        self,
        instance_execution_service,
        mock_gemini_dense_embedding_client,
        mock_vector_normalizer,
    ):
        """
        Verifies that embed_KB_document_dense correctly batches 250 documents
        into calls of 100, 100, and 50.
        """
        total_docs = 250
        dummy_docs = [Document(page_content=f"doc_{i+1}") for i in range(total_docs)]

        # Grabs the instance mock returned by 'mock_gemini_dense_embedding_client' rather than the class mock
        mock_client_instance = mock_gemini_dense_embedding_client.return_value

        # Returns list of dummy vectors matching the length of the input batch (e.g., [[1.0, 1.0], ...])
        mock_client_instance.embed_documents.side_effect = lambda texts: [[1.0, 1.0] for _ in texts]

        embedder = GeminiEmbedder(instance_execution_service)

        # Swap the 'doc_client' with configured mock
        embedder.doc_client = mock_client_instance

        results = embedder.embed_KB_document_dense(dummy_docs)

        assert len(results) == 250

        # Check that the API was called exactly 3 times
        assert mock_client_instance.embed_documents.call_count == 3

        # Verify the batch sizes
        calls = mock_client_instance.embed_documents.call_args_list

        # Call 1: Docs 1-100
        batch_1 = calls[0][0][0]  # args[0]
        assert len(batch_1) == 100
        assert batch_1[0] == "doc_1"

        # Call 2: Docs 101-200
        batch_2 = calls[1][0][0]
        assert len(batch_2) == 100
        assert batch_2[0] == "doc_101"

        # Call 3: Docs 200-250 (Remainder)
        batch_3 = calls[2][0][0]
        assert len(batch_3) == 50
        assert batch_3[-1] == "doc_250"


class TestGeminiEmbedderRetry:
    """Tests for rate limit retry behavior in embed_KB_document_dense."""

    @patch("knowledge_base.processing.gemini_embedder.time.sleep")
    @patch("knowledge_base.processing.gemini_embedder.retry_with_backoff")
    def test_resource_exhausted_triggers_retry(
        self,
        mock_retry,
        mock_sleep,
        instance_execution_service,
        mock_gemini_dense_embedding_client,
        mock_vector_normalizer,
    ):
        """Verifies that a RESOURCE_EXHAUSTED error triggers retry_with_backoff instead of raising immediately."""
        mock_client_instance = mock_gemini_dense_embedding_client.return_value
        mock_client_instance.embed_documents.side_effect = GoogleGenerativeAIError(
            "429 RESOURCE_EXHAUSTED"
        )
        mock_retry.return_value = [[1.0, 1.0]]

        embedder = GeminiEmbedder(instance_execution_service)
        embedder.doc_client = mock_client_instance

        results = embedder.embed_KB_document_dense([Document(page_content="test")])

        assert len(results) == 1
        mock_retry.assert_called_once()
        # Verify retry was called with the correct retryable exception type
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["retryable_on"] == (GoogleGenerativeAIError,)
        assert call_kwargs.kwargs["max_retries"] == 6

    @patch("knowledge_base.processing.gemini_embedder.time.sleep")
    def test_non_resource_exhausted_error_raises(
        self,
        mock_sleep,
        instance_execution_service,
        mock_gemini_dense_embedding_client,
        mock_vector_normalizer,
    ):
        """Verifies that a GoogleGenerativeAIError without RESOURCE_EXHAUSTED propagates immediately."""
        mock_client_instance = mock_gemini_dense_embedding_client.return_value
        mock_client_instance.embed_documents.side_effect = GoogleGenerativeAIError(
            "400 INVALID_ARGUMENT"
        )

        embedder = GeminiEmbedder(instance_execution_service)
        embedder.doc_client = mock_client_instance

        with pytest.raises(GoogleGenerativeAIError, match="INVALID_ARGUMENT"):
            embedder.embed_KB_document_dense([Document(page_content="test")])

    @patch("knowledge_base.processing.gemini_embedder.time.sleep")
    def test_generic_exception_raises_runtime_error(
        self,
        mock_sleep,
        instance_execution_service,
        mock_gemini_dense_embedding_client,
        mock_vector_normalizer,
    ):
        """Verifies that a non-GoogleGenerativeAIError exception is wrapped in RuntimeError."""
        mock_client_instance = mock_gemini_dense_embedding_client.return_value
        mock_client_instance.embed_documents.side_effect = ConnectionError("network failure")

        embedder = GeminiEmbedder(instance_execution_service)
        embedder.doc_client = mock_client_instance

        with pytest.raises(RuntimeError, match="network failure"):
            embedder.embed_KB_document_dense([Document(page_content="test")])
