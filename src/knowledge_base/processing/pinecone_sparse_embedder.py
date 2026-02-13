import logging
import time
from core.execution_service import ExecutionService
from knowledge_base.processing.text_chunker import TextChunker
from knowledge_base.processing.retry import retry_with_backoff
from knowledge_base.processing.vector_normalizer import VectorNormalizer, VectorType
from pinecone.core.openapi.inference.model.sparse_embedding import SparseEmbedding
from pinecone.exceptions import PineconeApiException
from pathlib import Path
from constants import PROCESSED_DATA_DIR, PINECONE_MAX_BATCH_SIZE
from typing import Literal, Union, List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PineconeSparseEmbedder:
    def __init__(self, execution_service: ExecutionService):
        """
        Initializes the PineconeSparseEmbedder with a Pinecone client.

        Args:
            execution_service (ExecutionService) : The service factory used to create
                configured clients.
        """
        self.pinecone_client = execution_service.get_pinecone_client()

    def embed_KB_document_sparse(self, inputs: Union[List[Document], List[str]]) -> List[SparseEmbedding]:
        """
        Generates normalized sparse embeddings for knowledge base documents.

        Delegates to _create_embeddings with task_type="passage", which handles
        batching, proactive throttling, and rate limit retry logic internally.

        Args:
            inputs (Union[List[Document], List[str]]): A list of LangChain Document
                objects or raw strings to embed.

        Returns:
            List[SparseEmbedding]: A list of normalized sparse embedding vectors.

        Raises:
            RuntimeError: If a batch fails with a non-retryable error or
                exhausts all retry attempts.
        """
        raw_embeddings = self._create_embeddings(task_type="passage", inputs=inputs)
        return VectorNormalizer.normalize(raw_embeddings, VectorType.SPARSE)  # type: ignore

    def embed_sparse_query(self, user_query: str) -> List[SparseEmbedding]:
        """
        Generates a normalized sparse embeddings for a user's search query.
        """

        raw_embeddings = self._create_embeddings(task_type="query", inputs=[user_query])
        return VectorNormalizer.normalize(raw_embeddings, VectorType.SPARSE)  # type: ignore

    def _create_embeddings(
        self,
        task_type: Literal["query", "passage"],
        inputs: Union[List[str], List[Document]],
        model_name: str = "pinecone-sparse-english-v0",
        max_tokens: int = 2048,
    ) -> List[SparseEmbedding]:
        """
        Internal helper to create sparse embeddings using Pinecone's inference API.

        Rate Limiting:
            A proactive 0.5s throttle is applied between batches to spread load.
            If a 429 (rate limit) response is received, the batch is retried with
            exponential backoff (max_retries=6, initial_delay=2, max_delay=60).
            Non-429 PineconeApiExceptions propagate immediately.

        Args:
            task_type (Literal["query", "passage"]): The type of embedding task.
            inputs (Union[List[str], List[Document]]): List of texts or Document objects to embed.
            model_name (str): The Pinecone model name to use for embedding.
            max_tokens (int): Maximum tokens per sequence for embedding.

        Returns:
            List[SparseEmbedding]: A list of sparse embedding vectors.

        Raises:
            RuntimeError: If a batch fails with a non-retryable error or
                exhausts all retry attempts.

        Example Return Object:
            [{'vector_type': 'sparse', 'sparse_values': [...], 'sparse_indices': [...]}, ...]
        """
        if all(isinstance(doc, Document) for doc in inputs):
            texts = [doc.page_content for doc in inputs]  # type: ignore
        else:
            texts = inputs

        all_embeddings = []
        # Ceiling division: rounds up so partial batches are counted (e.g., 101 texts / 96 batch size = 2 batches)
        total_batches = (len(texts) + PINECONE_MAX_BATCH_SIZE - 1) // PINECONE_MAX_BATCH_SIZE

        for batch_num, batch in enumerate(self._batch_texts(texts, PINECONE_MAX_BATCH_SIZE), 1):
            try:
                batch_embeddings = self.pinecone_client.inference.embed(
                    model=model_name,
                    inputs=batch,
                    parameters={
                        "input_type": task_type,
                        "max_tokens_per_sequence": max_tokens,
                        "truncate": "NONE",
                    },
                )
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Batch {batch_num}/{total_batches} complete.")
            except PineconeApiException as e:
                if e.status == 429:
                    logger.warning(f"Batch {batch_num}/{total_batches}: Hit Pinecone rate limit. Retrying with exponential backoff.")
                    batch_embeddings = retry_with_backoff(
                        fn = lambda: self.pinecone_client.inference.embed(
                            model=model_name,
                            inputs=batch,
                            parameters={
                                "input_type": task_type,
                                "max_tokens_per_sequence": max_tokens,
                                "truncate": "NONE",
                            }),
                        max_retries = 6,
                        initial_delay = 2,
                        max_delay = 60,
                        retryable_on = (PineconeApiException,)
                    )
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"Batch {batch_num}/{total_batches} complete (after retry).")
                else:
                    raise
            except Exception as e:
                error_msg = f"Error generating sparse embeddings with Pinecone for batch {batch_num}/{total_batches}: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
            time.sleep(0.5)

        return all_embeddings

    def _batch_texts(self, texts: List[str], batch_size: int):
        """Splits the list of texts into smaller batches."""
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]


# Example Usage
if __name__ == "__main__":  # pragma: no cover
    try:
        execution_service = ExecutionService()
        embedder = PineconeSparseEmbedder(execution_service)

        file_name = "Graduate-Student-Handbook-2024-2025.md"
        file_path = Path(PROCESSED_DATA_DIR) / file_name

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            chunker = TextChunker()
            chunks = chunker.split_text(markdown_content, "Graduate-Student-Handbook-2024-2025.md")

            # Test with 1 chunk to verify small batch logic works now
            small_test_chunks = chunks[50:51]
            print(small_test_chunks)
            embeddings = embedder.embed_KB_document_sparse(small_test_chunks)

            print(f"\nSuccess! Generated {len(embeddings)} embeddings.")
            if embeddings:
                print(f"Embeddings Type: {type(embeddings)}")
                print(embeddings)
                first_emb = embeddings[0]
                print(
                    f"First Embedding Type: {type(first_emb)}"
                )  # Expected Type:  <class 'pinecone.core.openapi.inference.model.sparse_embedding.SparseEmbedding'>
                print(f"First Embedding Content Preview: {first_emb}")
                print(len(first_emb.sparse_values))
                print(len(first_emb.sparse_indices))

    except Exception as e:
        logger.error(f"Test run failed: {e}")
