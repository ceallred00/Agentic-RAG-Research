import logging
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

logger = logging.getLogger(__name__)

def create_vector_db_index(pinecone_client: Pinecone, index_name: str):
    """
    Creates a vector database index using environment variables for configuration.
    """
    pc = pinecone_client

    try:
        pc.create_index(
            name = index_name, 
            vector_type = "dense", 
            dimension = 768, # Typical dimension for Gemini embeddings
            metric = "dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
        logger.info(f"Vector DB index '{index_name}' is ready.")

    except Exception as e:
        error_msg = f"Error creating vector database index: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    index = pc.Index(index_name)
    return index
