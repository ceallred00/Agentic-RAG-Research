import logging
import os
import json
from pathlib import Path
from enum import Enum
from typing import List, Optional, Union

from core.execution_service import ExecutionService
from core.logging_setup import setup_logging # Used for main entry
from knowledge_base.ingestion.pdf_to_markdown_converter import PDFToMarkdownConverter
from knowledge_base.processing.text_chunker import TextChunker
from knowledge_base.processing.pinecone_sparse_embedder import PineconeSparseEmbedder
from knowledge_base.processing.gemini_embedder import GeminiEmbedder
from knowledge_base.vector_db.create_vector_db_index import create_vector_db_index
from knowledge_base.vector_db.upsert_to_vector_db import upsert_to_vector_db
from langchain_core.documents import Document

from constants import PROCESSED_DATA_DIR, UWF_PUBLIC_KB_PROCESSED_DATE_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)

class SourceType(Enum):
    PDF = "pdf"
    MARKDOWN = "md"

    @property
    def extension(self) -> str:
        """Returns the file extension for this type."""
        return f".{self.value}"

class KnowledgeBasePipeline:
    def __init__(self, kb_name: str, raw_data_path: Union[Path, str] = RAW_DATA_DIR, processed_data_path: Union[Path, str] = PROCESSED_DATA_DIR):
        """
        Initializes the pipeline and all necessary services.

        Scenario: 
            PDF -> MD -> Vectors -> KB:
                raw_data_path should be the path to the directory storing the raw data files.
                processed_data_path is the path to which the converted MD files will be stored.
            
            MD -> Vectors -> KB:
                processed_data_path is the path from which the markdown files will be extracted.
        """
        self.kb_name = kb_name

        # Sanitize user input if string object is passed
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        
        self.execution_service = ExecutionService()
        
        self.converter = PDFToMarkdownConverter(
            raw_data_path = self.raw_data_path,                 # Reads from here
            processed_data_path = self.processed_data_path      # Saves to here
        )
        
        self.chunker = TextChunker()

        self.gemini_embedder = GeminiEmbedder(execution_service = self.execution_service)
        self.pinecone_embedder = PineconeSparseEmbedder(execution_service=self.execution_service)
        self.pc = self.pinecone_embedder.pinecone_client

        logger.info(f"KnowledgeBasePipeline initialized for KB: {self.kb_name}")

    def run(
        self,
        source_type: SourceType,
        specific_files: Optional[List[str]] = None
    ):
        """
        Main orchestrator method for ingesting documents into the Vector DB.
        1. Discovers files.
        2. Converts PDFs to Markdown (if needed).
        3. Chunks text.
        4. Embeds and Upserts to Pinecone.
        
        Args:
            source_type: Enum (SourceType.PDF or SourceType.MARKDOWN).
            specific_files: Optional list of filenames. If None, scans entire dir.
        """
        logger.info(f"Starting Pipeline for KB: {self.kb_name}")
        
        if source_type == SourceType.PDF:
            search_dir = self.raw_data_path
            logger.info(f"Source Type is PDF. Scanning RAW directory: {search_dir}")
        
        elif source_type == SourceType.MARKDOWN:
            search_dir = self.processed_data_path
            logger.info(f"Source Type is Markdown. Scanning PROCESSED directory: {search_dir}")
        
        else:
            raise ValueError(f"Unsupported SourceType: {source_type}")

        try:
            files = self._discover_files(source_dir=search_dir, source_type = source_type, specific_files= specific_files)
        except (ValueError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Pipeline execution stopped: {e}")
            raise
        
        except Exception as e:
            logger.error(f"Critical error during file discovery: {e}", exc_info = True)
            raise

        all_chunks = []
        skipped_md_files = []
        error_files = []

        for file_path in files:
            try:
                # If MD file already exists, returns file_path
                # Otherwise, it will load the PDF from self.raw_data_path arg and save to self.processed_data_path arg.
                md_path = self._ensure_markdown_exists(file_path = file_path, source_type=source_type)

                if not md_path:
                    logger.warning(f"Skipping {file_path.name} due to conversion failure.")
                    skipped_md_files.append(file_path)
                    continue
                
                try:
                    with open(md_path, 'r', encoding = 'utf-8') as f:
                        text_content = f.read()
                    
                    file_chunks = self.chunker.split_text(text = text_content, source_name = md_path.name)

                    if not file_chunks:
                        logger.warning(f"No chunks generated for file: {md_path.name}")
                        continue
                    
                    # File chunks already include enriched content and all necessary metadata (id, version, content, enriched content, etc.)
                    all_chunks.extend(file_chunks)
                
                except Exception as e:
                    logger.error(f"Error processing text/chunking for {md_path.name}: {e}", exc_info=True)
                    error_files.append(md_path.name)
                    continue
            
            except Exception as e:
                # Continue to iterate through list
                logger.error(f"Failed to process file {file_path.name}: {e}", exc_info = True)
        
        if all_chunks:
            self._export_chunks(all_chunks)
        else:
            error_msg = "Pipeline finished with no chunks generated. Exiting"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if skipped_md_files:
            logger.warning(f"{len(skipped_md_files)} files were skipped. File names: {skipped_md_files}")
        if error_files:
            logger.warning(f"Errors processing {len(error_files)} files. File names: {error_files}")
        
        logger.info(f"Successfully chunked {len(files)-len(skipped_md_files)-len(error_files)} files out of {len(files)} total files.")
        logger.info(f"Total chunks: {len(all_chunks)}")

        self._embed_and_upsert(all_chunks)

        logger.info("Pipeline completed successfully.")
        return all_chunks

    def _discover_files(self, source_dir: Path, source_type: SourceType, specific_files: Optional[List[str]]) -> List[Path]:
        """
        Locates and validates input files for the ingestion pipeline.

        This method acts as the gatekeeper for the pipeline. It can operate in two modes:
        1. Specific Mode: Validates a provided list of filenames within the source directory.
        2. Discovery Mode: Scans the source directory for all files matching the 
           provided `SourceType` extension.

        Args:
            source_dir (Path): The directory path to scan for input documents.
            source_type (SourceType): The Enum member indicating the target file 
                format (e.g., PDF or MARKDOWN).
            specific_files (Optional[List[str]]): An explicit list of filenames to 
                process. If None, the method performs a full directory scan.

        Returns:
            List[Path]: A list of validated absolute or relative Path objects 
                ready for processing.

        Raises:
            FileNotFoundError: If the `source_dir` does not exist, or if 
                `specific_files` contains filenames that cannot be found.
            ValueError: If no valid files are discovered (prevents downstream 
                service calls on empty data).
            PermissionError: If the application lacks read permissions for the 
                `source_dir`.
            Exception: Re-raises any unexpected errors encountered during 
                filesystem interaction.
        """
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        
        try:
            files: List[Path] = []
            
            # If the user passed in a list of specific files
            if specific_files:
                missing_files = []
                for filename in specific_files:
                    file_path = source_dir / filename
                    if file_path.exists():
                        files.append(file_path)
                    else:
                        missing_files.append(filename)
                
                if missing_files:
                    error_msg = f"Failed to find {len(missing_files)} requested files: {missing_files}."
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

            # Directory scan
            else: 
                # Look for any file with the correct extension
                files = list(source_dir.glob(f"*{source_type.extension}")) # Can raise PermissionError
            
            if not files:
                error_msg = (
                    f"No {source_type.name} files found in {source_dir}. "
                    "Pipeline cannot continue without input data."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
                        
            logger.info(f"Scanning {source_dir}. Found {len(files)} {source_type.name} files.")
            return files
        except PermissionError:
            logger.error(f"Permission denied accessing directory: {source_dir}")
            raise # Re-raise error because pipeline can't continue
        except Exception as e:
            logger.warning(f"Unexpected error during file discovery: {e}", exc_info=True)
            raise # Re-raise error bc pipeline can't continue
    
    def _ensure_markdown_exists(self, file_path: Path, source_type: SourceType) -> Optional[Path]:
        """
        Ensures a valid Markdown version of the input file is available for processing.

        This method coordinates the transformation of raw input into the required Markdown 
        format. It implements a 'lazy conversion' strategy to optimize performance:
        1. Markdown Passthrough: If the source is already Markdown, it returns the 
           original path.
        2. Cache Check: If the source is a PDF, it checks `processed_data_path` for 
           an existing `.md` file to avoid redundant conversion.
        3. PDF Conversion: If no cached version exists, it invokes the `PDFToMarkdownConverter` 
           to generate, save, and return the path to a new Markdown file.

        Args:
            file_path (Path): The path to the source file (PDF or MD).
            source_type (SourceType): The Enum member indicating if the file is 
                a PDF or MARKDOWN.

        Returns:
            Optional[Path]: The Path to the Markdown version of the document (either 
                the original file or the converted output). Returns None if the 
                conversion process fails or the document is empty.

        Note:
            Failures in conversion are logged as errors, but do not raise exceptions. 
            This allows the parent pipeline to skip problematic files and continue 
            processing the rest of the batch.
        """
        md_filename = f"{file_path.stem}.md"
        md_path = self.processed_data_path / md_filename

        if source_type == SourceType.MARKDOWN:
            return file_path

        if source_type == SourceType.PDF:
            if md_path.exists():
                logger.info(f"Skipping conversion (MD exists): {md_filename}")
                return md_path

            try:
                # Converter will load the PDF file from the directory located at the "raw_data_path" arg passed during KnowledgeBasePipeline instantiation
                doc = self.converter.load_pdf_as_markdown(file_path.name)

                if not doc:
                    return None
                
                # load_pdf_as_markdown returns a list of documents
                # no issue if length is one
                content = "\n\n".join([d.page_content for d in doc])

                # Saves PDF to the directory located at the "processed_data_path" arg passed during KnowledgeBasePipeline instantiation
                self.converter.save_markdown_file(content = content, output_filename = md_filename)
                logger.info(f"Converted {file_path} file to Markdown.")
                return md_path
            
            except Exception as e:
                logger.error(f"Conversion failed: {e}", exc_info = True)
                return None
    def _embed_and_upsert(self, chunks: List[Document]) -> None:
        """
        Generates hybrid embeddings and upserts vectors into the Pinecone index.

        This is a critical-path operation that orchestrates the final stage of the pipeline:
        1. Index Check: Verifies the target Pinecone index exists; creates it if absent.
           This runs first to fail fast before incurring embedding API costs.
        2. Dense Embedding: Generates dense vectors via Gemini (batched internally).
        3. Sparse Embedding: Generates sparse vectors via Pinecone inference (batched internally).
        4. Upsert: Writes the hybrid vectors and metadata to the Pinecone index.

        Args:
            chunks (List[Document]): Document chunks with enriched content and metadata,
                as produced by the chunking stage.

        Raises:
            RuntimeError: If dense or sparse embedding generation fails.
            Exception: If index creation or the upsert operation fails.

        """
        # Ensure Pinecone index exists before spending time/cost on embeddings
        try:
            if not self.pc.has_index(self.kb_name):
                logger.info(f"Pinecone index '{self.kb_name}' not found. Creating...")
                create_vector_db_index(pinecone_client=self.pc, index_name=self.kb_name)
            else:
                logger.info(f"Pinecone index '{self.kb_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to verify/create Pinecone index '{self.kb_name}': {e}")
            raise 

        # Generate dense embeddings (Gemini)
        try:
            logger.info(f"Generating dense embeddings for {len(chunks)} chunks...")
            dense_embeddings = self.gemini_embedder.embed_KB_document_dense(document=chunks)
            logger.info(f"Generated {len(dense_embeddings)} dense embeddings.")
        except RuntimeError as e:
            logger.error(f"Dense embedding failed for KB '{self.kb_name}'. Aborting pipeline.")
            raise 

        # Generate sparse embeddings (Pinecone)
        try:
            logger.info(f"Generating sparse embeddings for {len(chunks)} chunks...")
            sparse_embeddings = self.pinecone_embedder.embed_KB_document_sparse(inputs=chunks)
            logger.info(f"Generated {len(sparse_embeddings)} sparse embeddings.")
        except RuntimeError as e:
            logger.error(f"Sparse embedding failed for KB '{self.kb_name}'. Aborting pipeline.")
            raise 
        try:
            upsert_to_vector_db(
                pinecone_client=self.pc,
                index_name=self.kb_name,
                text_chunks=chunks,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
            )

            logger.info(f"Embed and upsert complete for KB: {self.kb_name}")
        except Exception as e:
            logger.error(f"Upsertion failed for KB '{self.kb_name}'. Aborting pipeline.")
            raise

    def _export_chunks(self, chunks: List[Document], filename: str = "all_chunks.json"):
            """
            Exports the generated chunks to a JSON file for inspection/backup.
            """
            # self.processed_data_path is a Path object (sanitized upon class initialization)
            output_path = self.processed_data_path / filename
            logger.info(f"Exporting {len(chunks)} chunks to {output_path}...")
            
            try:
                # Convert Document objects to dicts
                data_to_save = [
                    {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks
                ]

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                    
                logger.info("Export complete.")
            except Exception as e:
                logger.error(f"Failed to export chunks: {e}")

if __name__ == "__main__": # pragma: no cover
    setup_logging()
    pipeline = KnowledgeBasePipeline(kb_name = "uwf-kb-1",
                          processed_data_path = UWF_PUBLIC_KB_PROCESSED_DATE_DIR, raw_data_path= RAW_DATA_DIR)
    # chunks = pipeline.run(
    #                       source_type = SourceType.MARKDOWN,
    #                       specific_files = ["Advising_Syllabus.md", "Viewing_a_Degree_Audit.md"]
    #                       )
    # truncated_chunks = chunks[0:5] + chunks[40:45]

    # for chunk in truncated_chunks:
    #         print(f"\n\nChunk ID: {chunk.metadata.get('id')}")
    #         print(f"Chunk Metadata: {chunk.metadata}")
    #         print(f"\n{chunk.page_content}")

    #files = ['Phishing_Defense_Test_Simulation.md', 'Eduroam_Setup_Instructions_Chromebook.md', 'Eduroam_Setup_Instructions_Windows_11.md', 'How_to_Create_a_Profile_on_the_RecWell_Fusion_Portal_and_Sign_the_Release_of_Liability_Form_Community_Members.md', 'Eduroam_Setup_Instructions_macOS.md']
    
    chunks = pipeline.run(
        source_type = SourceType.MARKDOWN,
        specific_files = ["Zoom_Phone_Billing_Guidelines.md"]
    )

    