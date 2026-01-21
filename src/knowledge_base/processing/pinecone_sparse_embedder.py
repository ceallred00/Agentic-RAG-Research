import logging
from core.execution_service import ExecutionService
from knowledge_base.processing.text_chunker import TextChunker
from pinecone.core.openapi.inference.model.sparse_embedding import SparseEmbedding
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
        Generates sparse embeddings for knowledge base documents.
        """
        return self._create_embeddings(task_type = "passage", inputs=inputs)
    def embed_sparse_query(self, user_query:str):
        """
        Generates sparse embeddings for a user's search query.
        """

        return self._create_embeddings(task_type = "query", inputs=[user_query])

    def _create_embeddings(self, 
                           task_type: Literal["query", "passage"], 
                           inputs: Union[List[str], List[Document]], 
                           model_name: str = "pinecone-sparse-english-v0", 
                           max_tokens: int = 2048) -> List[SparseEmbedding]:
        """
        Internal helper to create sparse embeddings using Pinecone's inference API.
        
        Args:
            task_type (Literal["query", "passage"]): The type of embedding task.
            input (Union[List[str], List[Document]]): List of texts or Document objects to embed.
            model_name (str): The Pinecone model name to use for embedding.
            max_tokens (int): Maximum tokens per sequence for embedding.
        
        Returns:
            List[SparseEmbedding]: A list of sparse embedding vectors.
        
        Example Return Object:
            [{'vector_type': 'sparse', 'sparse_values': [...], 'sparse_indices': [...]}, ...]
        """
        if all(isinstance(doc, Document) for doc in inputs):
            texts = [doc.page_content for doc in inputs] #type: ignore
        else:
            texts = inputs
        
        all_embeddings = []
        for batch in self._batch_texts(texts, PINECONE_MAX_BATCH_SIZE):
            try: 
                batch_embeddings = self.pinecone_client.inference.embed(
                    model=model_name, 
                    inputs=batch,
                    parameters={"input_type": task_type, "max_tokens_per_sequence": max_tokens, "truncate": "NONE"}
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                error_msg = f"Error generating sparse embeddings with Pinecone for batch: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
        
        return all_embeddings
        
    def _batch_texts(self, texts: List[str], batch_size: int):
        """Splits the list of texts into smaller batches."""
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]
    
# Example Usage
if __name__ == "__main__":
    try:
        execution_service = ExecutionService()
        embedder = PineconeSparseEmbedder(execution_service)

        file_name = "Graduate-Student-Handbook-2024-2025.md"
        file_path = Path(PROCESSED_DATA_DIR) / file_name

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            
            chunker = TextChunker()
            chunks = chunker.split_text(markdown_content)
            
            # Test with just 2 chunks to verify small batch logic works now
            small_test_chunks = chunks[:2]
            embeddings = embedder.embed_KB_document_sparse(small_test_chunks)
            
            print(f"\nSuccess! Generated {len(embeddings)} embeddings.")
            if embeddings:
                print(f"Embeddings Type: {type(embeddings)}")
                print(embeddings)
                first_emb = embeddings[0]
                print(f"First Embedding Type: {type(first_emb)}")
                print(f"First Embedding Content Preview: {first_emb}")
                print(f"First Embedding: {first_emb.sparse_values}")

    except Exception as e:
        logger.error(f"Test run failed: {e}")
