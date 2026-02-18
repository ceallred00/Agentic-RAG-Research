import logging
import time
from typing import List, Union
from langchain_core.documents import Document
from langchain_google_genai._common import GoogleGenerativeAIError # Used for batch retry
from core.execution_service import ExecutionService
from knowledge_base.processing.vector_normalizer import VectorNormalizer, VectorType
from knowledge_base.processing.retry import retry_with_backoff
from constants import (
    GEMINI_EMBEDDING_BATCH_LIMIT,
    PROCESSED_DATA_DIR,
    GEMINI_EMBEDDING_MAX_CHAR_LIMIT,
)  # PROCESSSED_DATA_DIR and GEMINI_EMBEDDING_MAX_CHAR_LIMIT used for example usage

# Imports for the example usage:
from knowledge_base.processing.text_chunker import TextChunker
from pathlib import Path

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    def __init__(self, execution_service: ExecutionService):
        """
        Initializes the GeminiEmbedder with specialized clients for documents and queries.

        Args:
            execution_service (ExecutionService) : The service factory used to create
                configured LangChain clients.
        """
        # Create a client specifically for Document (Retriever side)
        self.doc_client = execution_service.get_embedding_client(
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
        )

        # Create a client specifically for queries (User side)
        self.query_client = execution_service.get_embedding_client(
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
        )

    def embed_KB_document_dense(self, document: Union[List[Document], str]) -> List[List[float]]:
        """
        Generates embeddings for a knowledge base document or a list of documents.

        This method uses the 'RETRIEVAL_DOCUMENT' task type, which optimizes
        the vector for storage and later retrieval. It automatically handles
        batching to respect the API limit of 100 documents per request.

        Rate Limiting:
            A proactive 0.5s throttle is applied between batches to spread load
            and minimize 429 errors. If a RESOURCE_EXHAUSTED error is still raised,
            the batch is retried with exponential backoff (max_retries=6,
            initial_delay=2, max_delay=60). Backoff delays: 2s, 4s, 8s, 16s, 32s,
            60s. This ensures the final retry waits long enough for Gemini's
            per-minute quota window (3,000 requests/min paid tier) to reset.

        Args:
            document (Union[List[Document], str]): A single string content
                or a list of LangChain Document objects to embed.

        Returns:
            List[List[float]]: A list of embedding vectors (list of floats).
                Even if a single string is passed, it returns a list containing
                one vector.

        Raises:
            RuntimeError: If a batch fails with a non-retryable error or
                exhausts all retry attempts.
        """
        embedding_model = self.doc_client
        if isinstance(document, str):
            texts = [document]
        else:
            texts = [doc.page_content for doc in document]

        raw_embeddings = []

        # Ceiling division: rounds up so partial batches are counted (e.g., 101 texts / 100 batch size = 2 batches)
        total_batches = (len(texts) + GEMINI_EMBEDDING_BATCH_LIMIT - 1) // GEMINI_EMBEDDING_BATCH_LIMIT
        logger.info(f"Starting dense embedding for {len(texts)} texts in {total_batches} batches of {GEMINI_EMBEDDING_BATCH_LIMIT}.")

        for batch_num, batch in enumerate(self._batch_texts(texts, GEMINI_EMBEDDING_BATCH_LIMIT), 1):
            try:
                batch_embeddings = embedding_model.embed_documents(batch)
                raw_embeddings.extend(batch_embeddings)
                logger.info(f"Batch {batch_num}/{total_batches} complete.")
            except GoogleGenerativeAIError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(f"Batch {batch_num}/{total_batches}: Hit Gemini rate limit. Retrying with exponential backoff.")
                    batch_embeddings = retry_with_backoff(
                        fn = lambda: embedding_model.embed_documents(batch),
                        max_retries = 6,
                        initial_delay = 2,
                        max_delay = 60,
                        retryable_on = (GoogleGenerativeAIError,)
                    )
                    raw_embeddings.extend(batch_embeddings)
                    logger.info(f"Batch {batch_num}/{total_batches} complete (after retry).")
                else:
                    raise
            except Exception as e:
                logger.error(f"Error generating dense embeddings for batch {batch_num}/{total_batches}: {e}")
                raise RuntimeError(f"Error generating dense embeddings for batch {batch_num}/{total_batches}: {e}") from e
            time.sleep(0.5)

        logger.info(f"Generated {len(raw_embeddings)} raw dense embeddings for KB document.")

        normalized_embeddings = VectorNormalizer.normalize(raw_embeddings, VectorType.DENSE)
        logger.info(f"Normalized {len(normalized_embeddings)} dense embeddings for KB document.")

        return normalized_embeddings  # type: ignore

    # TODO: Add error handling here.

    def embed_dense_query(self, query: str) -> List[float]:
        """
        Generates embeddings for a user search query.

        This method uses the 'RETRIEVAL_QUERY' task type, which optimizes
        the vector to find matching documents in the vector space.

        Safeguard:
            Inputs > GEMINI_EMBEDDING_MAX_CHAR_LIMIT characters in length are truncated to prevent API errors.

        Args:
            query (str): The search text provided by the user.

        Returns:
            List[float]: A single embedding vector representing the query.

        """
        if len(query) > GEMINI_EMBEDDING_MAX_CHAR_LIMIT:
            logger.warning(f"Query too long. Truncating to {GEMINI_EMBEDDING_MAX_CHAR_LIMIT} characters.")
            query = query[:GEMINI_EMBEDDING_MAX_CHAR_LIMIT]

        embedding_model = self.query_client
        raw_embeddings = embedding_model.embed_query(query)
        logger.info(f"Generated {len(raw_embeddings)} dense embeddings for query: {query}")

        normalized_batch = VectorNormalizer.normalize([raw_embeddings], VectorType.DENSE)
        logger.info(f"Normalized {len(normalized_batch)} dense embeddings for query.")
        return normalized_batch[0]  # type: ignore

    # TODO: Move to separate function to reduce redundant code between embedders.

    def _batch_texts(self, texts: List[str], batch_size: int):
        """Helper to split a list of texts into smaller batches."""
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]


if __name__ == "__main__":  # pragma: no cover
    execution_service = ExecutionService()
    embedder = GeminiEmbedder(execution_service)

    file_name = "Graduate-Student-Handbook-2024-2025.md"
    file_path = Path(PROCESSED_DATA_DIR) / file_name

    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        chunker = TextChunker()
        # Returns List[Document]
        chunks = chunker.split_text(markdown_content, "Graduate-Student-Handbook-2024-2025.md")
        # small_text_chunk = chunks[50:51]

    embeddings = embedder.embed_KB_document_dense(document=chunks)
    print(f"Success! Generated {len(embeddings)} embeddings.")
    print(embeddings[200][0:10])
