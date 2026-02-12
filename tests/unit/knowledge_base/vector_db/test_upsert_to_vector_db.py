import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from knowledge_base.vector_db.upsert_to_vector_db import upsert_to_vector_db


class TestUpsertToVectorDb:
    """Unit tests for the upsert_to_vector_db function."""

    def test_successful_upsert_to_vector_db(
        self,
        mock_pinecone_client,
        mock_index_object,
        text_chunk_with_metadata,
        normalized_dense_embeddings,
        normalized_sparse_embeddings,
        expected_record,
    ):
        """
        Verifies the 'Happy Path' where inputs are valid and the upsert succeeds.

        Checks:
        1. Correct Index Selection: The client connects to the specified index name.
        2. Data Transformation: Chunks, dense vectors, and sparse vectors are correctly
           zipped and formatted into the specific dictionary structure Pinecone requires.
        3. API Call: The index.upsert() method is called exactly once with the correct payload.
        """
        mock_pinecone_client.Index.return_value = mock_index_object

        upsert_to_vector_db(
            pinecone_client=mock_pinecone_client,
            index_name="fake_index_name",
            text_chunks=text_chunk_with_metadata,
            dense_embeddings=normalized_dense_embeddings,
            sparse_embeddings=normalized_sparse_embeddings,
        )

        mock_pinecone_client.Index.assert_called_once_with("fake_index_name")

        mock_index_object.upsert.assert_called_once_with(vectors=expected_record)

    def test_index_connection_exception(
        self,
        mock_pinecone_client,
        text_chunk_with_metadata,
        normalized_dense_embeddings,
        normalized_sparse_embeddings,
    ):
        """
        Verifies that if the client fails to connect to the Index (e.g., index not found
        or network error), the function raises the exception immediately before
        processing any data.
        """
        mock_pinecone_client.Index.side_effect = Exception("Error connecting to index")

        with pytest.raises(Exception) as exc_info:
            upsert_to_vector_db(
                pinecone_client=mock_pinecone_client,
                index_name="fake_index_name",
                text_chunks=text_chunk_with_metadata,
                dense_embeddings=normalized_dense_embeddings,
                sparse_embeddings=normalized_sparse_embeddings,
            )

        assert "Error connecting to index" in str(exc_info.value)

    def test_upsert_exception(
        self,
        mock_pinecone_client,
        mock_index_object,
        text_chunk_with_metadata,
        normalized_dense_embeddings,
        normalized_sparse_embeddings,
    ):
        """
        Verifies that runtime errors occurring during the actual batch upload
        (inside the loop) are caught, logged, and re-raised to the caller.
        """
        mock_pinecone_client.Index.return_value = mock_index_object
        mock_index_object.upsert.side_effect = Exception("Error upserting batch.")

        with pytest.raises(Exception) as exc_info:
            upsert_to_vector_db(
                pinecone_client=mock_pinecone_client,
                index_name="fake_index_name",
                text_chunks=text_chunk_with_metadata,
                dense_embeddings=normalized_dense_embeddings,
                sparse_embeddings=normalized_sparse_embeddings,
            )

        assert "Error upserting batch" in str(exc_info.value)

    @patch("knowledge_base.vector_db.upsert_to_vector_db.PINECONE_UPSERT_MAX_BATCH_SIZE", 2)
    def test_batches_large_payloads(self, mock_pinecone_client, mock_index_object):
        """
        Verifies that large payloads are split into smaller batches.
        """
        chunks = [Document(page_content=f"{i}", metadata={"id": f"{i}"}) for i in range(3)]
        dense = [[0.1] * 768] * 3  # List of 3 lists

        sparse_mock = MagicMock()
        sparse_mock.sparse_indices = [1]
        sparse_mock.sparse_values = [0.5]
        sparse = [sparse_mock] * 3

        mock_pinecone_client.Index.return_value = mock_index_object

        upsert_to_vector_db(
            pinecone_client=mock_pinecone_client,
            index_name="test_index",
            text_chunks=chunks,
            dense_embeddings=dense,
            sparse_embeddings=sparse,
        )

        # Batch size is 2. Total items is 3.
        # Expect 2 calls (Batch A=[1,2], Batch B=[3])
        assert mock_index_object.upsert.call_count == 2
    def test_list_dimension_mismatch(self, mock_pinecone_client, caplog):
        size = 3
        chunks = [Document(page_content=f"{i}", metadata={"id": f"{i}"}) for i in range(size)]
        dense = [[0.1]*768]*size
        sparse_mock = MagicMock()
        sparse_mock.sparse_indices = [1]
        sparse_mock.sparse_values = [0.5]
        sparse = [sparse_mock]

        mock_pinecone_client.Index = MagicMock()

        with pytest.raises(ValueError) as excinfo:
            upsert_to_vector_db(
                pinecone_client = mock_pinecone_client,
                index_name = "test_index",
                text_chunks = chunks,
                dense_embeddings = dense,
                sparse_embeddings = sparse
            )

        mock_pinecone_client.Index.assert_not_called()
        assert "Dimension mismatch between text chunks and embedding vectors" in caplog.text
        assert "Text Chunk Length: 3" in caplog.text
        assert "Sparse Embedding Length: 1" in caplog.text
        assert "Dimension mismatch between text chunks and embeddings." in str(excinfo.value)


