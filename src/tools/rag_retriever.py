import logging
from pinecone.grpc import PineconeGRPC
from pinecone.exceptions import PineconeException
from typing import List, Any
from knowledge_base.processing.gemini_embedder import GeminiEmbedder
from knowledge_base.processing.pinecone_sparse_embedder import PineconeSparseEmbedder

logger = logging.getLogger(__name__)


class RagRetriever:
    """
    Shared retrieval engine for hybrid RAG search (dense + sparse)

    Used by PerformRagTool (formats results as string for LLM) and the
    evlauation system (builds structured RetrievalResult for metrics).
    """
    def __init__(self, dense_embedder, sparse_embedder, pc_client, index_name):
        self._dense_embedder = dense_embedder
        self._sparse_embedder = sparse_embedder
        self._pc_client = pc_client
        self._index_name = index_name
    def retrieve_RAG_matches(self, user_query: str, top_k_matches: int = 5) -> List[Any]:
        """
        Executes the Hybrid RAG pipeline for a given user query. 

        This method performs the following steps:
        1.  Embeds the user query into both Dense (semantic) and Sparse (keyword) vectors
            using the configured Gemini and Pinecone embedders.
        2.  Queries the Pinecone index using Hybrid Search (Dense + Sparse) to retrieve
            the top-k most relevant document chunks.

        Args:
            user_query (str): The natural language question or search phrase provided by the user.
                              Example: "What is the deadline for dropping a class?"
            top_k_matches (int): The number of matches to be returned by the Pinecone API.
                                 Defaults to 5.
        
        Returns:
            List[Any]: List of Pinecone ScoredVector Objects. 
                Example Format: 
                [
                    {
                    "id": "vec3",
                    "score": 0,
                    "metadata": {"text": "Context:\nSource: UWF Public Knowledge Base / ITS", "url": "..."}
                    },
                    {
                    "id": "vec2",
                    "score": 0.0800000429,
                    "metadata": {"text": "Context:\nSource: UWF Public Knowledge Base\n...", "url": "..."}
                    }
                ]
        Raises:
            PineconeException: Captures and logs any Pinecone Exceptions related to querying
                                the pinecone DB.
            Exception: Captures and logs and errors during embedding or unexpected errors during
                        querying.
        """
        try:
            dense_vector = self._dense_embedder.embed_dense_query(user_query)
            logger.info(f"Successfully generated dense embedding of length {len(dense_vector)}")

        except Exception as e:
            logger.error(f"Error generating dense embeddings: {e}")
            raise e

        try:
            sparse_vector = self._sparse_embedder.embed_sparse_query(user_query)

            curr_sparse = sparse_vector[0] if isinstance(sparse_vector, list) else sparse_vector
            logger.info(
                f"""Successfully generated sparse embeddings.
                Indices length: {len(curr_sparse.sparse_indices)},
                Values length: {len(curr_sparse.sparse_values)}"""
            )

        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {e}")
            raise e

        index = self._pc_client.Index(self._index_name)

        try:
            response = index.query(
                top_k=top_k_matches,
                vector=dense_vector,
                sparse_vector={
                    "indices": curr_sparse.sparse_indices,
                    "values": curr_sparse.sparse_values,
                },
                include_values=False,  # Does not return vector values
                include_metadata=True,  # Returns metadata for context
            )

            matches = response.matches

            if matches:
                logger.info(
                    f"Query to Pinecone index '{self._index_name}' successful. Retrieved {len(matches)} matches."
                )

            return matches
        
        except PineconeException as e:
            error_msg = f"Pinecone error querying Pinecone index '{self._index_name}': {e}" 
            logger.error(error_msg, exc_info=True)
            raise
        except Exception as e:
            error_msg = f"Unexpected error querying Pinecone index '{self._index_name}': {e}"
            logger.error(error_msg)
            raise
