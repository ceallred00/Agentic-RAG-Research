import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from knowledge_base.processing.gemini_embedder import GeminiEmbedder 


class TestGeminiEmbedder:
    def test_doc_client_configuration(self, instance_execution_service):
        pass
    def test_query_client_configuration(self, instance_execution_service):
        pass
    def test_embed_KB_document_dense_batching(self,
        instance_execution_service, 
        mock_gemini_dense_embedding_client, 
        mock_vector_normalizer
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
        batch_1 = calls[0][0][0] # args[0]
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