import logging
from typing import List, Union
from langchain_core.documents import Document
from core.execution_service import ExecutionService
from knowledge_base.processing.vector_normalizer import VectorNormalizer, VectorType
from constants import GEMINI_EMBEDDING_BATCH_LIMIT, PROCESSED_DATA_DIR, GEMINI_EMBEDDING_MAX_CHAR_LIMIT # PROCESSSED_DATA_DIR and GEMINI_EMBEDDING_MAX_CHAR_LIMIT used for example usage

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
            model_name = "gemini-embedding-001",
            task_type = "RETRIEVAL_DOCUMENT",
        )

        # Create a client specifically for queries (User side)
        self.query_client = execution_service.get_embedding_client(
            model_name ="gemini-embedding-001",
            task_type = "RETRIEVAL_QUERY",
        )

    def embed_KB_document_dense(self, document: Union[List[Document], str]) -> List[List[float]]:
        """
        Generates embeddings for a knowledge base document or a list of doucments. 

        This method uses the 'RETRIEVAL_DOCUMENT' task type, which optimizes
        the vector for storage and later retrieval. It automatically handles 
        batching to respect the API limit of 100 documents per request.

        Args:
            document (Union[List[Document]], str]): A single string content
                or a list of LangChain Document objects to embed.
        
        Returns:
            List[List[float]]: A list of embedding vectors (list of floats).
                Even if a single string is passed, it returns a list containing
                one vector.
        """
        embedding_model = self.doc_client
        if isinstance(document, str):
            texts = [document]
        else:
            texts = [doc.page_content for doc in document]
        
        raw_embeddings = []

        logger.info(f"Starting dense embedding for {len(texts)} texts in batches of {GEMINI_EMBEDDING_BATCH_LIMIT}.")

        for batch in self._batch_texts(texts, GEMINI_EMBEDDING_BATCH_LIMIT):
            try:
                batch_embeddings = embedding_model.embed_documents(batch)
                raw_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating dense embeddings for batch: {e}")
                raise RuntimeError(f"Error generating dense embeddings for batch: {e}") from e
        
        logger.info(f"Generated {len(raw_embeddings)} raw dense embeddings for KB document.")
        
        normalized_embeddings = VectorNormalizer.normalize(raw_embeddings, VectorType.DENSE)
        logger.info(f"Normalized {len(normalized_embeddings)} dense embeddings for KB document.")
        
        return normalized_embeddings #type:ignore
        

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
        return normalized_batch[0] #type:ignore
    def _batch_texts(self, texts: List[str], batch_size: int):
        """Helper to split a list of texts into smaller batches."""
        for i in range(0, len(texts), batch_size):
            yield texts[i: i+batch_size]


if __name__ == "__main__": # pragma: no cover
    execution_service = ExecutionService()
    embedder = GeminiEmbedder(execution_service)

    file_name = "Graduate-Student-Handbook-2024-2025.md"
    file_path = Path(PROCESSED_DATA_DIR) / file_name

    if file_path.exists():
        with open(file_path, "r", encoding = "utf-8") as f:
            markdown_content = f.read()
        
        chunker = TextChunker()
        # Returns List[Document]
        chunks = chunker.split_text(markdown_content, "Graduate-Student-Handbook-2024-2025.md")
        #small_text_chunk = chunks[50:51]
    
    embeddings = embedder.embed_KB_document_dense(document = chunks)
    print(f"Success! Generated {len(embeddings)} embeddings.")
    print(embeddings[200][0:10])
