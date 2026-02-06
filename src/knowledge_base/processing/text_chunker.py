import logging
import hashlib
import frontmatter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional, Tuple, Dict
from constants import PROCESSED_DATA_DIR, CHUNKING_SIZE, CHUNKING_OVERLAP
from pathlib import Path

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles splitting of text into chunks, supporting both:
    1. Scraped Markdown (with YAML frontmatter for hierarchy/url).
    2. PDF/Raw Markdown (uses filename for source context).
    
    This class prioritizies semantic boundaries by first splitting on Markdown headers,
    and then recursively splitting larger sections to fit context windows.
    
    The final chunks retain metadata about their header hierarchy and source file, injected in the content.
    """
    def __init__(self, chunk_size: int = CHUNKING_SIZE, chunk_overlap: int = CHUNKING_OVERLAP):
        """
        Initialize the text chunker with configuration settings. 
        
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # IMPORTANT - Headers should be specified top-down for the _enrich_metadata method to properly inject the hierarchy.
        self.headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6")
        ]

    def split_text(self, text: str, source_name: Optional[str] = None) -> List[Document]:
        """
        Primary entry point: Takes raw markdown text and returns fully processed chunks.

        Pipeline:
            1. Detects if text has YAML metadata or is raw text.
            2. Split by Markdown Headers (preserves structural context).
            3. Split larger sections by Character count (preserves token limits).
            4. Inject metadata back into the text content so the vector "sees" it.
        
        Args:
            text (str): The raw markdown text to be chunked.
            source_name (Optional[str]): The name of the file. Used for metadata and content enrichment.


        Returns a List[Documents]. Example format: 

        [Document(metadata={'Header 1': 'Intro', 'Header 2': 'History', 'Source': 'markdown_file.md', 'id': 'markdown_file_chunk_1'}, 
                  page_content='Context: Source: markdown_file > Intro > History\n---\n## Intro  \n## History  \nMarkdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9]'),
        Document(metadata={'Header 1': 'Intro', 'Header 2': 'Rise and divergence', 'Source': 'markdown_file.md', 'id': 'markdown_file_chunk_2'}, 
                 page_content='Context: Source: markdown_file > Intro > Rise and divergence\n---\n## Rise and divergence  \nAs Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for  \nadditional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks.'),
                ...]
        """

        try:
            logger.info("Starting text chunking...")
            # Parse YAML metadata. 
            post = frontmatter.loads(text) # Plain text returns empty metadata
            clean_text = post.content
            file_metadata = post.metadata # Dict of YAML keys (if any)

            header_splits = self._split_on_headers(clean_text)
            recursive_chunks = self._split_recursive(header_splits) # Smaller context window
            final_chunks = self._enrich_metadata(recursive_chunks, file_metadata, source_name)
            
            logger.info(f"Successfully split text into {len(final_chunks)} chunks.")
            return final_chunks

        except Exception as e:
            logger.error(f"Error during text chunking: {e}", exc_info=True) # Append full stack trace to the log message.
            raise e

    def _split_on_headers(self, text: str) -> List[Document]:
        """
        Internal method to split text based on Markdown headers.

        Returns a list of Document objects, with each Document object representing the content contained within that header structure. 

        Example:

        [Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar'}, page_content='# Foo  \n## Bar  \nHi this is Jim  \nHi this is Joe'),
        Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar', 'Header 3': 'Boo'}, page_content='### Boo  \nHi this is Lance'),
        Document(metadata={'Header 1': 'Foo', 'Header 2': 'Baz'}, page_content='## Baz  \nHi this is Molly')]
        
        """
        try:
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split_on,
                strip_headers = False
            )
            return markdown_splitter.split_text(text) 

        
        except Exception as e:
            logger.error(f"Failed to split markdown headers: {e}")
            # Return whole text as one document if markdown split fails.
            return [Document(page_content=text)]
    
    def _split_recursive(self, documents: List[Document]) -> List[Document]:
        """
        Internal method to further split documents that are still too large, 
        while preserving the metadata (headers) from the previous step.
        
        Returns a list of Document objects. 
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators = ["\n\n", "\n", " ", ""]
        )

        logger.info(f"Successfully split chunks into: {self.chunk_size} with {self.chunk_overlap} character overlap.")

        return text_splitter.split_documents(documents)
        
    def _enrich_metadata(self, documents: List[Document], yaml_meta: Dict, source_name: Optional[str] = None) -> List[Document]:
        """
        Iterates through chunks to:
        1. Inject header hierarchy into the content (Context).
        2. Inject source name into metadata and content.
        3. Generates a deterministic, unique ID for each chunk.

        Before: 
            Metadata: {'Header 1': 'Admissions', 'Header 2': 'GPA'}
            Content: "The minimum requirement is 3.0"
        
        After:
            Metadata: {'Header 1': 'Admissions', 'Header 2': 'GPA', 'Source': 'graduate-handbook', 'id': 'graduate_handbook_chunk_7'}
            Content: "Context: Source: graduate-handbook > Admissions > GPA \n The minimum requirement is 3.0"
        
        Returns:
            List[Document]: The enriched documents with updated metadata and content.
        """
        enriched_docs = []

        #TODO: Finish here
        if yaml_meta:
            source_title = yaml_meta.get('title', 'Unknown Title')
            source_parent = yaml_meta.get('parent', 'Unknown Parent')
            source_url = yaml_meta.get('url', None)

        
        # Create a human-readable source name if provided.
        readable_source = ""
        # Create a URL for the source if provided. Used in ID generation.
        clean_filename_for_id = ""
        if source_name:
            # Extract just the stem if a full filename or path is provided.
            source_name = Path(source_name).stem

            # Readable for semantic context.
            readable_source = source_name.replace("-", " ").replace("_", " ")

            # Cleaned for ID generation.
            clean_filename_for_id = source_name.replace(" ","_").replace("-","_").lower()
        
        for i, doc in enumerate(documents):
            context_parts = []
            
            if source_name:
                doc.metadata["source"] = readable_source

                chunk_id = f"{clean_filename_for_id}_chunk_{i+1}"

                context_parts.append(f"Source: {readable_source}")
            
            else:
                unique_string = f"{doc.page_content}-{i}"
                # Generates a 32-character string
                content_hash = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
                # Total ID length: 37 characters
                chunk_id = f"anon_{content_hash}"

            doc.metadata["id"] = chunk_id

            for header in self.headers_to_split_on:
                header_title = header[1]
                if header_title in doc.metadata:
                    context_parts.append(doc.metadata[header_title])
            
            if context_parts:
                context_str = " > ".join(context_parts)
                new_content =f"Context: {context_str}\n---\n{doc.page_content}"

                doc.page_content = new_content

            # Add doc to the list regardless if markdown headers existed.
            enriched_docs.append(doc)
    
        return enriched_docs
                

# --- Example Usage ---
if __name__=="__main__": # pragma: no cover
    file_name = "Graduate-Student-Handbook-2024-2025.md"
    file_path = Path(PROCESSED_DATA_DIR) / file_name

    if file_path.exists():
        with open(file_path, "r", encoding = "utf-8") as f:
            markdown_content = f.read()
        
        chunker = TextChunker()
        # Returns List[Document]
        chunks = chunker.split_text(markdown_content)
        print(type(chunks))
        print(type(chunks[0]))

        for i, chunk in enumerate(chunks[:100]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content Preview: {chunk.page_content}")
    else:
        print(f"File not found: {file_path}")



    
