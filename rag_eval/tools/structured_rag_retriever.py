import logging
from tools.rag_retriever import RagRetriever
from rag_eval.schemas.eval_schemas import RetrievalResult

logger = logging.getLogger(__name__)

class StructuredRagRetriever:
    """
    Thin adapter around RagRetriever that returns structured RetrievalResult
    objects instead of raw Pinecone ScoredVectors.

    Used by the evaluation system to access individual contexts, scores,
    metadata, and IDs for RAGAS metric computation.
    """

    def __init__(self, rag_retriever: RagRetriever):
        self._retriever = rag_retriever

    def retrieve(self, user_query: str, top_k_matches: int = 5) -> RetrievalResult:
        """
        Runs hybrid retrieval and maps raw Pinecone matches into a RetrievalResult.

        Args:
            user_query (str): The natural language query to retrieve against.
            top_k_matches (int): Number of top results to retrieve. Defaults to 5.

        Returns:
            RetrievalResult: Structured result containing contexts, scores,
                             metadata, and vector IDs for each match.
        """
        # Returns List[ScoredVector Objects]
        logger.info(f"Retrieving matches for query: {user_query}")
        matches = self._retriever.retrieve_RAG_matches(
            user_query= user_query,
            top_k_matches = top_k_matches
        )

        contexts = []
        metadatas = []
        scores = []
        ids = []

        for match in matches:
            # Pinecone's ScoredVector objects use attribute access (dot notation)
            ids.append(match.id)
            scores.append(match.score)
            metadatas.append(match.metadata)
            contexts.append(match.metadata.get("text", ""))

        return RetrievalResult(
            question = user_query,
            contexts = contexts,
            metadata = metadatas,
            scores = scores,
            ids = ids
        )
