import pytest
from unittest.mock import MagicMock
from pinecone.exceptions import PineconeException, PineconeApiException

class TestRagRetriever:
    """Tests for RagRetriever.retrieve_RAG_matches: hybrid search, sparse format handling, and error propagation."""

    USER_QUERY = "When is the last day to drop classes?"
    TOP_K = 5

    def test_retriever_happy_path(self,
                                  normalized_dense_embeddings,
                                  normalized_sparse_embeddings,
                                  rag_retriever,
                                  mock_pinecone_matches,
                                  mock_index_object):
        """Successful hybrid retrieval returns Pinecone matches and queries with correct dense/sparse vectors."""
        # Arrange
        # Fixture returns a List[List[float]]
        # Grabbing first object to mock what would be returned by function call
        rag_retriever._dense_embedder.embed_dense_query.return_value = normalized_dense_embeddings[0]
        rag_retriever._sparse_embedder.embed_sparse_query.return_value = normalized_sparse_embeddings
        mock_index_object.query.return_value.matches = mock_pinecone_matches

        # Normalized sparse embeddings fixture should return as list
        expected_sparse = normalized_sparse_embeddings[0]

        # rag_retriever._pc_client.Index.return_value is mocked in mock_pc_client fixture
        # Act
        results = rag_retriever.retrieve_RAG_matches(user_query = self.USER_QUERY,
                                                     top_k_matches = self.TOP_K)

        mock_index_object.query.assert_called_once_with(top_k = self.TOP_K,
                                                          vector = normalized_dense_embeddings[0],
                                                          sparse_vector = {
                                                              "indices": expected_sparse.sparse_indices,
                                                              "values": expected_sparse.sparse_values,
                                                          },
                                                          include_values = False,
                                                          include_metadata=True)

        assert results == mock_pinecone_matches

    def test_dense_embedding_failure_logs_and_reraises(self,
                                                       normalized_dense_embeddings,
                                                       rag_retriever,
                                                       caplog):
        """Dense embedding failure is logged and re-raised; sparse embedding is never called."""
        # Arrange
        rag_retriever._dense_embedder.embed_dense_query.side_effect = RuntimeError("Dense embedding failed")

        # Act
        with pytest.raises(RuntimeError):
            rag_retriever.retrieve_RAG_matches(user_query = self.USER_QUERY,
                                               top_k_matches = self.TOP_K)

        # Assert
        # Should not reach this
        rag_retriever._sparse_embedder.embed_sparse_query.assert_not_called()
        assert "Error generating dense embeddings" in caplog.text

    @pytest.mark.parametrize("exception", [
        PineconeApiException,
        RuntimeError
    ])
    def test_sparse_embedding_failure_logs_and_reraises(self,
                                                        exception,
                                                        normalized_dense_embeddings,
                                                        rag_retriever,
                                                        mock_index_object,
                                                        caplog):
        """Sparse embedding failure (PineconeApiException or RuntimeError) is logged and re-raised; Pinecone query is never called."""
        # Arrange
        rag_retriever._dense_embedder.embed_dense_query.return_value = normalized_dense_embeddings[0]
        rag_retriever._sparse_embedder.embed_sparse_query.side_effect = exception

        # Act/Assert
        with pytest.raises(exception):
            rag_retriever.retrieve_RAG_matches(user_query = self.USER_QUERY,
                                               top_k_matches = self.TOP_K)

        # Assert
        mock_index_object.query.assert_not_called()
        assert "Error generating sparse embeddings" in caplog.text

    @pytest.mark.parametrize("exception, caplog_text", [
        [PineconeException, "Pinecone error querying Pinecone index"],
        [Exception, "Unexpected error querying Pinecone index"]
    ])
    def test_query_exception_handling(self, exception, caplog_text, normalized_dense_embeddings,
                                normalized_sparse_embeddings,
                                rag_retriever,
                                mock_index_object,
                                caplog):
        """PineconeException and generic Exception from index.query() are logged with distinct messages and re-raised."""
        # Arrange
        rag_retriever._dense_embedder.embed_dense_query.return_value = normalized_dense_embeddings[0]
        rag_retriever._sparse_embedder.embed_sparse_query.return_value = normalized_sparse_embeddings
        mock_index_object.query.side_effect = exception

        # Act/Assert
        with pytest.raises(exception):
            rag_retriever.retrieve_RAG_matches(user_query = self.USER_QUERY,
                                               top_k_matches = self.TOP_K)

        # Assert
        assert caplog_text in caplog.text

    @pytest.mark.parametrize("as_list", [True, False])
    def test_handles_sparse_vector_format(self,
                                          as_list,
                                          normalized_sparse_embeddings,
                                          normalized_dense_embeddings,
                                          rag_retriever,
                                          mock_index_object,
                                          mock_pinecone_matches):
        """Sparse embedder returning a list or a single object both produce the same Pinecone query."""
        # Arrange
        # Fixture returns list
        sparse_input = normalized_sparse_embeddings if as_list else normalized_sparse_embeddings[0]

        rag_retriever._dense_embedder.embed_dense_query.return_value = normalized_dense_embeddings[0]
        rag_retriever._sparse_embedder.embed_sparse_query.return_value = sparse_input
        mock_index_object.query.return_value.matches = mock_pinecone_matches

        # If as_list - curr_sparse should grab the first element
        # else - curr_sparse is set to the single object returned
        # Either case - It should be the first argument of the list
        expected_curr_sparse = normalized_sparse_embeddings[0]

        response = rag_retriever.retrieve_RAG_matches(
            user_query = self.USER_QUERY,
            top_k_matches = self.TOP_K
        )

        mock_index_object.query.assert_called_once_with(
            top_k = self.TOP_K,
            vector = normalized_dense_embeddings[0],
            sparse_vector = {
                "indices": expected_curr_sparse.sparse_indices,
                "values": expected_curr_sparse.sparse_values,
            },
            include_values = False,
            include_metadata = True
        )
