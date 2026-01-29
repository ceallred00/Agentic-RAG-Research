import pytest
from unittest.mock import MagicMock, patch
from knowledge_base.vector_db.create_vector_db_index import create_vector_db_index

class TestCreateVectorDbIndex:
    """Unit tests for the create_vector_db_index function"""
    @patch("knowledge_base.vector_db.create_vector_db_index.ServerlessSpec")
    def test_successful_db_creation(self, MockServerlessSpec, mock_pinecone_client, mock_index_object):
        """
        Verifies that the function initializes the ServerlessSpec correctly, 
        calls create_index with the expected configuration, and returns the
        resulting index object.
        """
        mock_pinecone_client.Index.return_value = mock_index_object

        mock_spec_instance = MagicMock()
        MockServerlessSpec.return_value = mock_spec_instance

        result = create_vector_db_index(mock_pinecone_client, "fake_index_name")

        MockServerlessSpec.assert_called_once_with(cloud="aws", region="us-east-1")

        mock_pinecone_client.create_index.assert_called_once_with(
            name = "fake_index_name",
            vector_type = "dense",
            dimension = 768,
            metric = "dotproduct",
            spec = mock_spec_instance
        )

        assert result == mock_index_object

    def test_vector_db_exception(self, mock_pinecone_client):
        """
        Verifies that exceptions raised by the Pinecone client during creation
        are caught, logged, and re-raised as RuntimeErrors with the original message.
        """
        mock_pinecone_client.create_index.side_effect = RuntimeError("Something went wrong")

        with pytest.raises(RuntimeError) as exc_info:
            result = create_vector_db_index(mock_pinecone_client, "fake_index_name")
        
        assert "Something went wrong" in str(exc_info.value)
