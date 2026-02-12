import logging
import hashlib
import re
import frontmatter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from typing import List, Optional, Tuple, Dict
from constants import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR, # Used in example usage
    CHUNKING_SIZE,
    CHUNKING_OVERLAP,
    UWF_PUBLIC_KB_PROCESSED_DATE_DIR,
)
from pathlib import Path

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Handles splitting of text into chunks, supporting both:
    1. Scraped Markdown (with YAML frontmatter for hierarchy/url/versioning).
    2. PDF/Raw Markdown (uses filename for source context).

    This class prioritizes semantic boundaries by first splitting on Markdown headers,
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


        Returns:
            List[Documents].

        Example PDF Return Format:
            [Document(metadata={'Header 1': 'Intro', 'Header 2': 'History', 'Source': 'markdown_file.md', 'id': 'markdown_file_chunk_1'},
                    page_content='Context: Source: markdown_file > Intro > History\n---\n## Intro  \n## History  \nMarkdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9]'),
            Document(metadata={'Header 1': 'Intro', 'Header 2': 'Rise and divergence', 'Source': 'markdown_file.md', 'id': 'markdown_file_chunk_2'},
                    page_content='Context: Source: markdown_file > Intro > Rise and divergence\n---\n## Rise and divergence  \nAs Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for  \nadditional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks.'),
                    ...]

        Example Confluence URL Return Format:

        """

        try:
            logger.info("Starting text chunking...")
            # Parse YAML metadata.
            post = frontmatter.loads(text)  # Plain text returns empty metadata
            clean_text = post.content
            file_metadata = post.metadata

            # Split on headers
            header_splits = self._split_on_headers(clean_text)

            # Split on chunk size (smaller context window)
            recursive_chunks = self._split_recursive(header_splits)

            # Metadata Injection & ID Generation
            final_chunks = self._enrich_metadata(recursive_chunks, file_metadata, source_name)

            logger.info(f"Successfully split text into {len(final_chunks)} chunks.")
            return final_chunks

        except Exception as e:
            logger.error(f"Error during text chunking: {e}", exc_info=True)
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
                strip_headers=False,  # Keep headers in content so context isn't lost
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
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        logger.info(f"Successfully split chunks into: {self.chunk_size} with {self.chunk_overlap} character overlap.")

        return text_splitter.split_documents(documents)

    def _enrich_metadata(
        self,
        documents: List[Document],
        yaml_meta: Dict,
        source_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Enriches document chunks with context, metadata, and unique IDs for RAG retrieval.

        This method iterates through the provided documents to:
        1. Construct a rich 'Context' string (Source > Version > Headers) and prepend it to the content for the LLM.
        2. Preserve the original, clean text in metadata for UI display.
        3. Generate a pre-formatted 'breadcrumbs' string for frontend citations.
        4. Assign a deterministic, unique ID to each chunk to prevent duplicates.

        Example 1: Confluence Page (Rich Metadata)
        ------------------------------------------
        Input:
            Metadata: {'Header 1': 'Campus Resources', 'Header 2': 'Weather Emergency Information'}
            Content:  "## Weather Emergency Information\n* WUWF-FM (88.1MHz) is the official information source for the University..."
            YAML Meta: {'title': 'Advising Syllabus',
                        'parent': 'Academic Advising',
                        'path': 'UWF Public Knowledge Base / Academic Adivisng / Advising Syllabus',
                        'original_url': ''https://confluence.uwf.edu/pages/viewpage.action?pageId=42669534',
                        'page_id': 42669534,
                        'version': '34',
                        'last_updated': 'datetime.datetime(2022, 4, 12, 15, 55, 51, 613000, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=68400)))'
                        }

        Output:
            Metadata: {
                'Header 1': 'Campus Resources',
                'Header 2': 'Weather Emergency Information',
                'id': '42669534_chunk_40',
                'source': 'Advising Syllabus',
                'url': 'https://confluence.uwf.edu/pages/viewpage.action?pageId=42669534',
                'parent': 'Academic Advising',
                'breadcrumbs': 'Source: UWF Public Knowledge Base / Academic Advising / Advising Syllabus | Version: 34 | Last Updated: 2022-04-12 | Headers: Campus Resources > Weather Emergency Information'
                'version': 34,
                'last_updated': '2022-04-12',
                'original_content': "## Weather Emergency Information \n",
                'text': "Context:...\n##Weather Emergency Information \n..." (Enriched)
            }
            Content:
                "Context:
                 Source: UWF Public Knowledge Base / Academic Advising / Advising Syllabus
                 Version: 34
                 Last Updated: 2022-04-12
                 Headers: Campus Resources > Weather Emergency Information
                 ---
                 ## Weather Emergency Information \n..."

        Example 2: Raw PDF/Text File (Inferred Metadata)
        ------------------------------------------------
        Input:
            Metadata: {
                        'Header 1': 'THE DEPARTMENT OF MATHEMATICS AND STATISTICS',
                        'Header 2': 'Program Overview'
                    }
            Content:  "## Program Overview  \nThe Master of Science in Mathematical Sciences..."
            Source Name: "Graduate-Student-Handbook-2024-2025.md"

        Output:
            Metadata: {
                'Header 1': 'THE DEPARTMENT OF MATHEMATICS AND STATISTICS',
                'Header 2': 'Program Overview',
                'id': 'graduate_student_handbook_2024_2025_chunk_18',
                'source': 'Graduate Student Handbook 2024 2025',
                'url': '',
                'parent': 'Document Library',
                'breadcrumbs': 'Source: Document Library / Graduate Student Handbook 2024 2025 | Headers: THE DEPARTMENT OF MATHEMATICS AND STATISTICS > Program Overview',
                'version': None,
                'last_updated': None,
                'text': 'Context:\n...\n---\n## Program Overview  \nThe Master of Science in Mathematical Sciences...'
                'original_content': '## Program Overview  \nThe Master of Science in Mathematical Sciences...'
            }
            Content:
                Context:
                Source: Document Library / Graduate Student Handbook 2024 2025
                Headers: THE DEPARTMENT OF MATHEMATICS AND STATISTICS > Program Overview
                ---
                ## Program Overview
                The Master of Science in Mathematical Sciences...

        Returns:
            List[Document]: The enriched documents ready for vector embedding.
        """
        enriched_docs = []

        # Defaults
        doc_title = "Unknown Document"
        doc_parent = ""
        doc_url = ""
        doc_root_id = ""  # Used for creating deterministic chunk IDs
        doc_path_string = ""

        # Versioning defaults
        doc_version = None
        doc_last_updated = None
        doc_last_updated_readable = None

        # Confluence data
        if yaml_meta:
            doc_title = yaml_meta.get("title", doc_title)
            doc_parent = yaml_meta.get("parent", "")
            doc_url = yaml_meta.get("original_url", "")
            doc_path_string = yaml_meta.get("path", "")  # "UWF Public KB / Parent / Page"

            # Extract versioning info
            doc_version = yaml_meta.get("version", None)
            doc_last_updated = yaml_meta.get("last_updated", None)

            if doc_last_updated:
                doc_last_updated_readable = (
                    str(doc_last_updated).split("T")[0].split(" ")[0]
                )  # Handles Iso format and Python datetimes

            # Use Confluence Page ID as the root for chunk IDs (Very Stable)
            if "page_id" in yaml_meta:
                doc_root_id = str(yaml_meta["page_id"])

        # PDF / Raw File (Inferred Metadata)
        if not doc_root_id and source_name:
            # Clean filename: "Graduate-Handbook-2024.md" -> "Graduate Handbook 2024"
            stem = Path(source_name).stem
            doc_title = stem.replace("-", " ").replace("_", " ").title()
            doc_parent = "Document Library"  # Generic parent for files

            # Create a stable ID root from filename: "graduate_handbook_2024"
            doc_root_id = re.sub(r"[^a-zA-Z0-9]", "_", stem).lower()

        # Fallback ID
        if not doc_root_id:
            # Deterministic short ID
            # Hashing entire content for guaranteed uniqueness
            doc_root_id = "anon_" + hashlib.md5(documents[0].page_content.encode()).hexdigest()[:8]

        for i, doc in enumerate(documents):
            # Generate ID
            # Format: {page_id_or_filename}_chunk_{index} - 'graduate_handbook_2024_chunk_1' or '42669534_chunk_1'
            chunk_id = f"{doc_root_id}_chunk_{i+1}"

            # Build context string (for LLM to read)
            context_parts = []

            # Add source hierarchy
            if doc_path_string:
                context_parts.append(f"Source: {doc_path_string}")  # Use full path if available
            elif doc_parent:
                context_parts.append(f"Source: {doc_parent} / {doc_title}")
            else:
                context_parts.append(f"Source: {doc_title}")

            # Add versioning (if available)
            if doc_version:
                context_parts.append(f"Version: {doc_version}")
            if doc_last_updated_readable:
                context_parts.append(f"Last Updated: {doc_last_updated_readable}")

            # Add header context (Extracted by MarkdownHeaderTextSplitter)
            header_context = []
            for _, header_name in self.headers_to_split_on:
                if header_name in doc.metadata:
                    header_context.append(doc.metadata[header_name])

            if header_context:
                context_parts.append(f"Headers: {' > '.join(header_context)}")

            context_str = "\n".join(context_parts)

            new_content = f"Context:\n{context_str}\n---\n{doc.page_content}"

            # Update metadata (For database filtering)
            doc.metadata.update(
                {
                    "id": chunk_id,
                    "source": doc_title,
                    "url": doc_url,
                    "parent": doc_parent,
                    "breadcrumbs": context_str.replace(
                        "\n", " | "
                    ),  # Easily returns source information to frontend (more customizable than raw enriched content)
                    "version": doc_version,
                    "last_updated": doc_last_updated_readable,
                    "text": new_content,  # Gives RAG LLM access to source path, header hierarchy, versioning info, date of last update, in addition to raw context.
                    "original_content": doc.page_content,  # Keep pure content for LLM to return to the user (doesn't require second DB lookup for content)
                }
            )

            doc.page_content = new_content
            enriched_docs.append(doc)

        return enriched_docs


if __name__ == "__main__":  # pragma: no cover
    chunker = TextChunker()

    # Define files to test
    files_to_process = [
        # PDF / Raw Markdown File
        {
            "name": "Graduate-Student-Handbook-2024-2025.md",
            "path": Path(PROCESSED_DATA_DIR) / "Graduate-Student-Handbook-2024-2025.md"
        },
        # Confluence Scraped File (YAML)
        # {
        #     "name": "Advising Syllabus",
        #     "path": Path(UWF_PUBLIC_KB_PROCESSED_DATE_DIR) / "Advising_Syllabus.md",
        # }
    ]

    print("\n=== STARTING CHUNKER TEST ===\n")

    for file_info in files_to_process:
        f_path = file_info["path"]
        f_name = file_info["name"]

        print(f"\n---> Processing: {f_name}")

        if f_path.exists():
            with open(f_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            # Run the chunker
            chunks = chunker.split_text(markdown_content, source_name=f_name)

            print(f"    Generated {len(chunks)} chunks.")

            truncated_chunks = chunks[20:25]
            print(truncated_chunks)

            # Print preview of the first chunk to verify metadata injection
            # if truncated_chunks:
            #     for chunk in truncated_chunks:
            #         print(f"\n\nChunk ID: {chunk.metadata.get('id')}")
            #         print(f"Chunk Metadata: {chunk.metadata}")
            #         print(f"\n{chunk.page_content}")
        else:
            print(f"    [ERROR] File not found at: {f_path}")

    print("\n=== TEST COMPLETE ===")
