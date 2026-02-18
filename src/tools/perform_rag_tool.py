import logging
from typing import Type, List, Any
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from tools.rag_retriever import RagRetriever
from core.execution_service import ExecutionService
from knowledge_base.processing.gemini_embedder import GeminiEmbedder
from knowledge_base.processing.pinecone_sparse_embedder import PineconeSparseEmbedder

logger = logging.getLogger(__name__)


class RagSearchInput(BaseModel):
    """Input schema for the RAG search tool."""

    user_query: str = Field(description="The natural language query to search for.")


class PerformRagTool(BaseTool):
    """
    Performs a RAG search over a knowledge base.
    Embeds the user query (dense + sparse) and retrieves relevant documents from the Pinecone index.
    """

    name: str = "perform_rag_search"
    description: str = """Searches the University of West Florida knowledge base for context relevant
    to the user's query using RAG. Use this to answer UWF-specific questions about policies, courses, or campus life."""
    args_schema: Type[BaseModel] = RagSearchInput

    # Dependency injection
    # Private attributes to hold retriever instances
    # Hidden from LLM serialization.
    _retriever: RagRetriever = PrivateAttr()

    def _run(self, user_query: str, top_k_matches: int = 5) -> str:
        """
        Delegates hybrid RAG retrieval to `RagRetriever` and formats the results
        into a context string for the LLM.

        Args:
            user_query (str): The natural language question or search phrase provided by the user.
                              Example: "What is the deadline for dropping a class?"
            top_k_matches (int): Number of top results to retrieve from Pinecone. Defaults to 5.

        Returns:
            str: A formatted string containing the content of the retrieved documents,
                 ready to be passed to the LLM as context.
        """
        logger.info(f"Tool '{self.name} processing query: '{user_query}'")
        matches = self._retriever.retrieve_RAG_matches(user_query = user_query, top_k_matches = top_k_matches)

        return self._format_results(matches)

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

    Creates a `RagRetriever` (with dense/sparse embedders and a Pinecone client)
    and injects it into the tool's private `_retriever` attribute. This Dependency
    Injection pattern is required because Pydantic's `__init__` (used by BaseTool)
    cannot accept complex service objects directly.

    Args:
        execution_service (ExecutionService): The central service factory used to create
                                              authenticated clients for Gemini and Pinecone.
        index_name (str): The name of the Pinecone index to be queried (e.g., "uwf-kb-1").

    Returns:
        PerformRagTool: A fully configured instance of the tool, ready to be bound
                        to an LLM or Agent.
    """
    tool = PerformRagTool()

    retriever = RagRetriever(
        dense_embedder = GeminiEmbedder(execution_service),
        sparse_embedder= PineconeSparseEmbedder(execution_service),
        pc_client = execution_service.get_pinecone_client(),
        index_name = index_name
    )

    tool._retriever = retriever

    return tool
