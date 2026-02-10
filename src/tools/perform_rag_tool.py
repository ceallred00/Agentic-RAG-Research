import logging
from typing import Type, List, Any
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

    def _run(self, user_query: str) -> str:
        """
        Executes the Hybrid RAG (Retrieval-Augmented Generation) pipeline for a given user query.

        This method performs the following steps:
        1.  Embeds the user query into both Dense (semantic) and Sparse (keyword) vectors
            using the configured Gemini and Pinecone embedders.
        2.  Queries the Pinecone index using Hybrid Search (Dense + Sparse) to retrieve
            the top 5 most relevant document chunks.
        3.  Formats the retrieved matches into a single context string using `_format_results`.

        Args:
            user_query (str): The natural language question or search phrase provided by the user.
                              Example: "What is the deadline for dropping a class?"

        Returns:
            str: A formatted string containing the content of the retrieved documents,
                 ready to be passed to the LLM as context.
                 Returns a "No matches found" message if retrieval fails or yields no results.

        Raises:
            Exception: Captures and logs any errors during embedding or retrieval, returning
                       a user-friendly error string to the LLM to prevent a crash.
        """
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
            logger.info(
                f"Successfully generated sparse embeddings. Indices length: {len(curr_sparse.sparse_indices)}, Values length: {len(curr_sparse.sparse_values)}"
            )

        except Exception as e:
            logger.error(f"Error generating sparse embeddings {self.name}: {e}")
            raise e

        index = self._pc_client.Index(self._index_name)

        try:
            response = index.query(
                top_k=5,
                vector=dense_vector,
                sparse_vector={
                    "indices": curr_sparse.sparse_indices,
                    "values": curr_sparse.sparse_values,
                },
                include_values=False,  # Does not return vector values
                include_metadata=True,  # Returns metadata for context
            )

            matches = response.matches  # type: ignore
            if matches:
                logger.info(
                    f"Query to Pinecone index '{self._index_name}' successful. Retrieved {len(matches)} matches."
                )

            return self._format_results(matches)

        except Exception as e:
            logger.error(
                f"Error querying Pinecone index '{self._index_name}': {e}",
                exc_info=True,
            )
            return f"Error performing RAG search: {e}"

    def _format_results(self, matches: List[Any]) -> str:
        """
        Formats a list of Pinecone search matches into a single, clean context string.

        This method iterates through the retrieved matches, extracts the 'text' metadata field
        (which contains the pre-processed headers, source, and content), and appends the
        retrieval score for transparency.

        Args:
            matches (List[Any]): A list of Pinecone `ScoredVector` objects returned from the index query.
                                 Each match is expected to have a `metadata` dictionary containing
                                 a 'text' key.

        Returns:
            str: A single string containing all retrieved document chunks, separated by
                 headers and newlines.

        Example Output:
            --- Result (Score: 0.89) ---
            Context:
            Source: UWF Public Knowledge Base / Registrar / Academic Calendar
            Headers: Fall 2024 > Important Dates
            ---
            # Drop/Add Period
            The last day to drop a class without a grade of 'W' is August 25th.

            --- Result (Score: 0.75) ---
            ... (Next Chunk) ...
        """
        context_parts = []

        for match in matches:
            meta = match.metadata or {}

            default_msg = "No content available."
            content = meta.get("text", default_msg)  # Ingestion pipeline stores enriched content in the 'text' field.

            if content == default_msg:
                logging.warning(f"No content available for vector: {meta.get('id', "Unknown ID")}")

            score = f"{match.score:.2f}" if match.score else "N/A"

            block = f"--- Result (Score: {score}) ---\n" f"{content}\n"

            context_parts.append(block)

        return "\n".join(context_parts)


# Factory Function
def get_perform_rag_tool(execution_service: ExecutionService, index_name: str) -> PerformRagTool:
    """
    Factory function to instantiate and configure the `PerformRagTool`.

    This function implements the Dependency Injection pattern to initialize the tool
    with complex services (ExecutionService, Embedders, Pinecone Client) that cannot
    be passed directly to the Pydantic `__init__` method.

    It creates a blank instance of `PerformRagTool` and manually injects the
    dependencies into its private attributes (`_dense_embedder`, `_pc_client`, etc.).

    Args:
        execution_service (ExecutionService): The central service factory used to create
                                              authenticated clients for Gemini and Pinecone.
        index_name (str): The name of the Pinecone index to be queried (e.g., "uwf-kb-v1").

    Returns:
        PerformRagTool: A fully configured instance of the tool, ready to be bound
                        to an LLM or Agent.
    """
    tool = PerformRagTool()

    tool._dense_embedder = GeminiEmbedder(execution_service)
    tool._sparse_embedder = PineconeSparseEmbedder(execution_service)
    tool._pc_client = execution_service.get_pinecone_client()
    tool._index_name = index_name

    return tool
