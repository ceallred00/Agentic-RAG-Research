import logging
from pathlib import Path
from core.execution_service import ExecutionService
from knowledge_base.ingestion.pdf_to_markdown_converter import PDFToMarkdownConverter
from knowledge_base.processing.text_chunker import TextChunker
from knowledge_base.processing.pinecone_sparse_embedder import PineconeSparseEmbedder
from knowledge_base.processing.gemini_embedder import GeminiEmbedder
from knowledge_base.vector_db.create_vector_db_index import create_vector_db_index
from constants import PROCESSED_DATA_DIR, RAW_DATA_DIR, PINECONE_UPSERT_MAX_BATCH_SIZE
from typing import List
from pinecone.grpc import PineconeGRPC as Pinecone

from knowledge_base.vector_db.upsert_to_vector_db import upsert_to_vector_db

logger = logging.getLogger(__name__)

def PDF_to_KB_Pipeline(pdf_file_names: List[str], kb_name: str):
    """
    Workflow:
        1. Checks for existing markdown files; converts PDFs if not present.
        2. Processes the markdown to extract text chunks.
        3. Generates sparse and dense embeddings.
        4. Checks for existing Pinecone index; creates if absent.
        5. Upserts embeddings into the Pinecone index.

    Args:
        pdf_file_names (List[str]): List of names of the PDF files.
        kb_name (str): Name of the knowledge base to create or update.

    Returns:
        None
    """
    # Initialize the PDF to Markdown converter.
    converter = PDFToMarkdownConverter(raw_data_path=RAW_DATA_DIR, processed_data_path=PROCESSED_DATA_DIR)
    
    # Initialize the embedding services.
    execution_service = ExecutionService()
    gemini_embedder = GeminiEmbedder(execution_service=execution_service)
    pinecone_embedder = PineconeSparseEmbedder(execution_service=execution_service)
    pc = pinecone_embedder.pinecone_client

    # Initialize the text chunker.
    chunker = TextChunker()
    all_chunks = []
    
    # Process each PDF file and save as markdown.
    for pdf_file_name in pdf_file_names:
        file_stem = Path(pdf_file_name).stem
        
        md_filename = f"{file_stem}.md"
        md_file_path = PROCESSED_DATA_DIR / md_filename

        if not md_file_path.exists():
            document = converter.load_pdf_as_markdown(pdf_file_name)
            converter.save_markdown_file(content = document[0].page_content, 
                                         output_filename = md_filename)
            logger.info(f"Completed conversion of PDFs to markdown for KB: {kb_name}")
        else:
            logger.info(f"Markdown file already exists: {md_filename}")
        
        with open(md_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        file_chunks = chunker.split_text(markdown_content, source_name=file_stem)
        all_chunks.extend(file_chunks)
        logger.info(f"Extracted {len(file_chunks)} chunks from {md_filename}")
    
    logger.info(f"Total chunks extracted for KB '{kb_name}': {len(all_chunks)}")

    for chunk in all_chunks[:10]:
        print(f"Metadata: {chunk.metadata}\nContent Preview: {chunk.page_content[:400]}\n---\n")

    # Create dense embeddings for the text chunks.
    dense_embeddings = gemini_embedder.embed_KB_document_dense(document=all_chunks)
    if dense_embeddings:
        logger.info(f"Generated {len(dense_embeddings)} dense embeddings for KB: {kb_name}")


    # Create sparse embeddings for the text chunks.
    sparse_embeddings = pinecone_embedder.embed_KB_document_sparse(inputs=all_chunks)
    if sparse_embeddings:
        logger.info(f"Generated {len(sparse_embeddings)} sparse embeddings for KB: {kb_name}")
    
    # Check if the Pinecone index exists; if not, create it.
    try:
        if not pc.has_index(kb_name):
            logger.info(f"Pinecone index '{kb_name}' does not exist. Creating new index.")
            create_vector_db_index(pinecone_client=pc, index_name=kb_name)
    
    except Exception as e:
        logger.error(f"Error checking/creating Pinecone index: {e}")
        raise e
    
    # Upsert the embeddings into the Pinecone index.
    upsert_to_vector_db(
        pinecone_client=pc,
        index_name=kb_name,
        text_chunks=all_chunks,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings
    )
    

if __name__ == "__main__":
    pdf_files = ["Graduate-Student-Handbook-2024-2025.pdf"]
    knowledge_base_name = "university-handbook-kb"
    PDF_to_KB_Pipeline(pdf_file_names=pdf_files, kb_name=knowledge_base_name)
            

    

    
    
    
    