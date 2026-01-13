import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple
from constants import PROCESSED_DATA_DIR
from pathlib import Path

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles the splitting of text document into smaller chunks for embedding.
    
    This class prioritizies semantic boundaries by first splitting on Markdown headers,
    and then recursively splitting larger sections to fit context windows."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker with configuration settings. 
        
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
        ]

    def split_text(self, text: str) -> List[Document]:
        """
        Primary entry point: Takes raw markdown text and returns fully processed chunks.

        Pipeline:
            1. Split by Markdown Headers (preserves structural context).
            2. Split larger sections by Character count (preserves token limits).

        Returns a List[Documents]. Example format: 

        [Document(metadata={'Header 1': 'Intro', 'Header 2': 'History'}, page_content='# Intro  \n## History  \nMarkdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9]'),
        Document(metadata={'Header 1': 'Intro', 'Header 2': 'History'}, page_content='Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files.'),
        Document(metadata={'Header 1': 'Intro', 'Header 2': 'Rise and divergence'}, page_content='## Rise and divergence  \nAs Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for  \nadditional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks.'),
        Document(metadata={'Header 1': 'Intro', 'Header 2': 'Rise and divergence'}, page_content='#### Standardization  \nFrom 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort.'),
        Document(metadata={'Header 1': 'Intro', 'Header 2': 'Implementations'}, page_content='## Implementations  \nImplementations of Markdown are available for over a dozen programming languages.')]
        """

        try:
            header_splits = self._split_on_headers(text)
            final_chunks = self._split_recursive(header_splits)

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

        return text_splitter.split_documents(documents)
    

# --- Example Usage ---
if __name__=="__main__":
    file_name = "Graduate-Student-Handbook-2024-2025.md"
    file_path = Path(PROCESSED_DATA_DIR) / file_name

    if file_path.exists():
        with open(file_path, "r", encoding = "utf-8") as f:
            markdown_content = f.read()
        
        chunker = TextChunker()
        chunks = chunker.split_text(markdown_content)

        for i, chunk in enumerate(chunks[:10]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content Preview: {chunk.page_content}")
    else:
        print(f"File not found: {file_path}")



    
