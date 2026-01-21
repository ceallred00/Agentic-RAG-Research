import logging
from typing import List, Optional, Type
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from knowledge_base.processing.gemini_embedder import GeminiEmbedder
from core.execution_service import ExecutionService

logger = logging.getLogger(__name__)

class EmbedQueryInput(BaseModel):
    """Input schema for the embedding tool."""
    user_query: str = Field(description="The text query to be converted into a vector embedding.")

class EmbedUserQueryTool(BaseTool):
    """
    Tool that converts a natural language query into a vector embedding.
    Useful for preparing a query before performing a vector similarity search against the knowledge base.
    """
    name: str = "embed_user_query"
    description: str = "Converts a text query into a list of floats (vector embeddings). This is a mandatory prerequisite for using the perform_rag tool "
    args_schema: Type[BaseModel] = EmbedQueryInput

    # Dependency injection
    _embedder: GeminiEmbedder = PrivateAttr() # Hidden from LLM

    def __init__(self, execution_service: ExecutionService):
        """
        Initialize the tool with the execution service.
        """
        super().__init__() # Calls BaseTool.__init__()
        self._embedder = GeminiEmbedder(execution_service)
    
    def _run(self, user_query: str) -> List[float]:
        """Synchronous execution of the tool."""
        logger.info(f"Tool '{self.name} processing query: '{user_query}'")
        try:
            vector = self._embedder.embed_dense_query(user_query)
            logger.info(f"Successfully generated embedding of length {len(vector)}")
            return vector
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            raise e

# Factory Function
def get_embed_query_tool(execution_service: ExecutionService) -> EmbedUserQueryTool:
    """Factory function to instantiate the tool with dependencies."""
    return EmbedUserQueryTool(execution_service=execution_service)



    
