import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from pinecone.grpc import PineconeGRPC as Pinecone
from constants import PINECONE_UPSERT_MAX_BATCH_SIZE

logger = logging.getLogger(__name__)

def upsert_to_vector_db(
    pinecone_client: Pinecone,
    index_name: str,
    text_chunks: List[Document],
    dense_embeddings: List[List[float]],
    sparse_embeddings: List[Any]
) -> None:
    """
    Upserts a batch of text chunks and their embeddings into a Hybrid Pinecone index.
    
    Args:
        pinecone_client: Authenticated Pinecone client.
        index_name: The name of the target index.
        text_chunks: List of Document objects containing text and metadata.
        dense_embeddings: List of dense vector embeddings corresponding to chunks.
        sparse_embeddings: List of sparse vector embeddings corresponding to chunks.
    
    Pinecone Hybrid Index Documentation: https://docs.pinecone.io/guides/search/hybrid-search#use-a-single-hybrid-index
    """
    
    try:
        index = pinecone_client.Index(index_name)
    except Exception as e:
        logger.error(f"Failed to connect to index '{index_name}': {e}")
        raise e

    records = []
    # Zip together the chunks and their corresponding embeddings
    for chunk, dense, sparse in zip(text_chunks, dense_embeddings, sparse_embeddings):
        
        chunk_id = chunk.metadata.get('id')
        if not chunk_id:
            logger.warning(f"Chunk missing 'id'. Source: {chunk.metadata.get('source', 'Unknown')}. Skipping.")
            continue

        # Remove redundant 'id' from metadata
        metadata_payload = chunk.metadata.copy()
        metadata_payload.pop('id', None)

        records.append({
            'id': chunk_id,
            'values': dense,
            'sparse_values': {
                'indices': sparse.sparse_indices, 
                'values': sparse.sparse_values
            },
            'metadata': {
                'text': chunk.page_content,
                **metadata_payload
            }
        })

    # Upsert in batches to avoid exceeding Pinecone size limits.
    total_vectors = len(records)
    if total_vectors == 0:
        logger.warning(f"No valid vectors to upsert to index '{index_name}'.")
        return

    logger.info(f"Starting upsert of {total_vectors} vectors to index '{index_name}'...")

    for i in range(0, total_vectors, PINECONE_UPSERT_MAX_BATCH_SIZE):
        batch = records[i : i + PINECONE_UPSERT_MAX_BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
            batch_num = i // PINECONE_UPSERT_MAX_BATCH_SIZE + 1
            logger.info(f"Upserted batch {batch_num} ({len(batch)} vectors).")
        except Exception as e:
            logger.error(f"Failed to upsert batch starting at index {i}: {e}")
            raise e

    logger.info(f"Successfully finished upserting {total_vectors} vectors.")