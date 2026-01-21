import logging
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from knowledge_base.processing.gemini_embedder import GeminiEmbedder
from knowledge_base.processing.pinecone_sparse_embedder import PineconeSparseEmbedder
from pinecone.grpc import PineconeGRPC
from core.execution_service import ExecutionService

logger = logging.getLogger(__name__)

class RagSearchInput(BaseModel):
    """Input schema for the RAG search tool."""
    user_query: str = Field(description="The natural language query to search for.")

class PerformRagTool(BaseTool):
    """
    Performs a RAG (Retrieval-Augmented Generation) search over a knowledge base.
    Embeds the user query (dense + sparse) and retrieves relevant documents from the Pinecone index.
    """
    name: str = "perform_rag_search"
    description: str = """Searches the University of West Florida knowledge base for context relevant 
    to the user's query using RAG. Use this to answer UWF-specific questions about policies, courses, or campus life."""
    args_schema: Type[BaseModel] = RagSearchInput

    # Dependency injection
    # Private attributes to hold the embedder instances, Pinecone client, and index name.
    # All are hidden from LLM serialization.
    _dense_embedder: GeminiEmbedder = PrivateAttr()
    _sparse_embedder: PineconeSparseEmbedder = PrivateAttr()
    _pc_client: PineconeGRPC = PrivateAttr()
    _index_name: str = PrivateAttr()

    def __init__(self, execution_service: ExecutionService, index_name: str):
        """
        Initialize the tool with the execution service.

        Args:
            execution_service (ExecutionService): The service factory to create configured clients.
            index_name (str): The name of the Pinecone index to search.
        """
        super().__init__() # Calls BaseTool.__init__()
        self._dense_embedder = GeminiEmbedder(execution_service)
        self._pc_client = execution_service.get_pinecone_client()
        self._index_name = index_name
        self._sparse_embedder = PineconeSparseEmbedder(execution_service)
    
    def _run(self, user_query: str) -> str:
        """Executes the RAG pipeline."""
        logger.info(f"Tool '{self.name} processing query: '{user_query}'")
        try:
            dense_vector = self._dense_embedder.embed_dense_query(user_query)
            logger.info(f"Successfully generated dense embedding of length {len(dense_vector)}")
        
        except Exception as e:
            logger.error(f"Error generating dense embeddings {self.name}: {e}")
            raise e
        
        try:
            sparse_vector = self._sparse_embedder.embed_sparse_query(user_query)

            curr_sparse = sparse_vector[0] if isinstance(sparse_vector, list) else sparse_vector
            logger.info(f"Successfully generated sparse embeddings. Indices length: {len(curr_sparse.sparse_indices)}, Values length: {len(curr_sparse.sparse_values)}")
        
        except Exception as e:
            logger.error(f"Error generating sparse embeddings {self.name}: {e}")
            raise e
        
        index = self._pc_client.Index(self._index_name)

        try:
            response = index.query(
                top_k = 5,
                vector = dense_vector,
                sparse_vector = {'indices': curr_sparse.sparse_indices, 
                                'values': curr_sparse.sparse_values},
                include_values = False, # Does not return vector values
                include_metadata = True # Returns metadata for context
            )

            matches = response.matches #type:ignore
            
            if not matches:
                logger.warning(f"No matches found for query: '{user_query}'")
                return "No relevant information found in the knowledge base."
            
            context_str = ""
            for match in matches:
                meta = dict(match.metadata or {})

                source = meta.pop('source', 'Unknown Source')
                text_snippet = meta.pop('text', 'No content available')

                chunk_block = f"Source: {source}\n"

                for key,value in meta.items():
                    chunk_block += f"{key}: {value}\n"
                
                chunk_block += f"Content: {text_snippet}\n---\n"
                context_str += chunk_block

            logger.info(f"Query to Pinecone index '{self._index_name}' successful. Retrieved {len(matches)} matches.")
            return context_str
        
        except Exception as e:
            logger.error(f"Error querying Pinecone index '{self._index_name}': {e}", exc_info=True)
            return f"Error performing RAG search: {e}"

        
# Factory Function
def get_perform_rag_tool(execution_service: ExecutionService, index_name: str) -> PerformRagTool:
    """Factory function to instantiate the tool with dependencies."""
    return PerformRagTool(execution_service=execution_service, index_name = index_name)