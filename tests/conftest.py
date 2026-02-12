"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import SecretStr
from pathlib import Path
from langchain_core.documents import Document
from pinecone.core.openapi.inference.model.sparse_embedding import SparseEmbedding


from schemas.agent_schemas import AgentConfig
from core.execution_service import ExecutionService
from knowledge_base.ingestion.pdf_to_markdown_converter import PDFToMarkdownConverter
from knowledge_base.ingestion.confluence_content_extractor import (
    ConfluenceContentExtractor,
)
from knowledge_base.ingestion.confluence_page_processor import ConfluencePageProcessor
from knowledge_base.ingestion.url_to_md_converter import URLtoMarkdownConverter
from knowledge_base.processing.text_chunker import TextChunker
from knowledge_base.pipeline.knowledge_base_pipeline import KnowledgeBasePipeline

# ==============================================================================
# 1. CONSTANTS & DATA FIXTURES
#    - Basic dictionaries and configuration data used across tests.
# ==============================================================================


@pytest.fixture
def base_agent_config():
    """
    Returns a base agent configuration dictionary.
    Configuration is modified in individual tests as needed.
    """
    return {
        "version": "1.0",
        "agent_metadata": {
            "name": "base_test_agent",
            "description": "A test agent configuration following valid schema.",
        },
        "model": {
            "provider": "google",
            "name": "gemini-3-pro-preview",
            "temperature": 0.5,
        },
        "system_prompt": "You are a helpful assistant.",
    }


@pytest.fixture
def sample_agent_config_dict():
    """Returns a dictionary containing multiple valid agent configurations."""
    return {
        "good_test_agent_1": {
            "version": "1.0",
            "agent_metadata": {
                "name": "good_test_agent_1",
                "description": "A test agent configuration following valid schema.",
            },
            "model": {
                "provider": "google",
                "name": "gemini-3-pro-preview",
                "temperature": 0.5,
            },
            "system_prompt": "You are a helpful assistant.",
        },
        "good_test_agent_2": {
            "version": "1.0",
            "agent_metadata": {
                "name": "good_test_agent_2",
                "description": "Another test agent configuration following valid schema.",
            },
            "model": {
                "provider": "google",
                "name": "gemini-3-small",
                "temperature": 0.7,
            },
            "system_prompt": "You are a creative assistant.",
        },
    }


@pytest.fixture
def valid_pdf_filename():
    """Returns a valid PDF filename for testing."""
    return "valid_test_document.pdf"


@pytest.fixture
def valid_pdf_filepath(valid_raw_data_dir, valid_pdf_filename):
    """Creates and returns the path to a valid PDF file in the raw data directory."""
    pdf_path = Path(valid_raw_data_dir / valid_pdf_filename)
    pdf_path.touch()  # Create an empty file for testing
    return pdf_path


@pytest.fixture
def valid_md_filename():
    """Returns a valid Markdown filename for testing."""
    return "valid_test_document.md"


@pytest.fixture
def valid_md_filepath(valid_processed_data_dir, valid_md_filename):
    """Creates and returns the path to a valid Markdown file in the processed data directory."""
    md_path = Path(valid_processed_data_dir / valid_md_filename)
    md_path.touch()  # Create an empty file for testing
    return md_path


@pytest.fixture
def valid_md_file_content():
    """
    Returns sample markdown content with sections long enough
    to force recursive splitting when chunk_size is small.
    """
    long_paragraph = "This is a long sentence meant to exceed the chunk limit."
    return (
        f"# Title\n"
        f"Intro text is short.\n"
        f"## Section 1\n"
        f"Details about section 1.{long_paragraph}\n"
        f"### Subsection A\n"
        f"Deep dive.{long_paragraph}\n"
        f"## Section 2\n"
        f"Conclusion."
    )


@pytest.fixture
def valid_md_file(valid_md_filepath, valid_md_file_content):
    """Creates a valid markdown file with sample content."""
    with open(valid_md_filepath, "w", encoding="utf-8") as f:
        f.write(valid_md_file_content)
    return valid_md_filepath


@pytest.fixture
def long_text_chunk():
    """Test chunk used to test the functionality of the TextChunker class"""
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text_doc = Document(page_content=text, metadata={"Header 1": "Test Header"})
    return [text_doc]


@pytest.fixture
def short_text_chunk():
    """Text chunk used to test the functionality of the TextChunker class"""
    short_text = "123456789"
    short_doc = Document(
        page_content=short_text,
        metadata={"Header 1": "First Header", "Header 2": "Second Header "},
    )
    return [short_doc]


@pytest.fixture
def text_chunk_with_metadata():
    """
    Returns a list containing a single Document object.
    The page_content is pre-formatted to match the output of TextChunker.split_text,
    including the injected context header and source metadata.
    """
    # TextChunker logic: Source filename is cleaned (stem only, underscores to spaces)
    # Source: "hello_world.pdf" -> "hello world"
    expected_context = "Context: Source: hello world > Test > Test Test\n---\nHello World"

    text_chunk = Document(
        page_content=expected_context,
        metadata={
            "id": "hello_world_chunk_1",
            "source": "hello_world.pdf",
            "Header 1": "Test",
            "Header 2": "Test Test",
        },
    )
    return [text_chunk]


@pytest.fixture
def example_test_chunk_from_handbook():
    return [
        Document(
            metadata={
                "Header 1": "THE DEPARTMENT OF MATHEMATICS AND STATISTICS",
                "Header 2": "Departmental Program Requirements",
                "source": "Graduate Student Handbook 2024 2025",
                "id": "graduate_student_handbook_2024_2025_chunk_51",
            },
            page_content="Context: Source: Graduate Student Handbook 2024 2025 > THE DEPARTMENT OF MATHEMATICS AND STATISTICS > Departmental Program Requirements\n---\n## Departmental Program Requirements  \nTo earn your Advanced Data Science Graduate Certificate online through UWF, you will complete a total of nine credit hours. The program consists of three required courses that provide specialized skills, leading to career advancement opportunities across various fields.",
        )
    ]


@pytest.fixture
def raw_dense_embeddings():
    """
    Returns a list containing a single dense vector embedding (list of floats).
    Matches the structure expected by the upsert function (List[List[float]]).
    """
    return [[2.0, 4.0, 4.0, 8.0]]


@pytest.fixture
def normalized_dense_embeddings():
    """
    Returns a list containing a single normalized dense vector embedding (list of floats).
    """
    return [[0.2, 0.4, 0.4, 0.8]]


@pytest.fixture
def raw_sparse_embeddings():
    """
    Returns a list containing a real SparseEmbedding object.
    Using the real class ensures we validate proper data types and attribute names.
    """
    # Create the actual object
    # Note: Pinecone's SparseEmbedding usually expects 'indices' and 'values'
    # Check your specific SDK version if it requires 'sparse_indices' instead.
    embedding = SparseEmbedding(
        sparse_values=[1.0, 5.0, 5.0, 7.0],
        sparse_indices=[744372458, 2165993515, 3261080123, 3508911095],
        vector_type="sparse",
    )

    return [embedding]


@pytest.fixture
def normalized_sparse_embeddings():
    """
    Returns a list containing a normalized SparseEmbedding object.
    """
    embedding = SparseEmbedding(
        sparse_values=[0.1, 0.5, 0.5, 0.7],
        sparse_indices=[744372458, 2165993515, 3261080123, 3508911095],
        vector_type="sparse",
    )

    return [embedding]


@pytest.fixture
def expected_record():
    """
    Returns the expected Pinecone record structure corresponding to the
    'text_chunk_with_metadata' fixture.

    The structure includes:
    - 'id': Extracted from the chunk metadata.
    - 'values': The single dense vector (float list) for this specific chunk.
    - 'sparse_values': Dictionary of indices and values.
    - 'metadata': Contains the page_content as 'text' plus original metadata (minus 'id').
    """
    return [
        {
            "id": "hello_world_chunk_1",
            "values": [0.2, 0.4, 0.4, 0.8],
            "sparse_values": {
                "indices": [744372458, 2165993515, 3261080123, 3508911095],
                "values": [0.1, 0.5, 0.5, 0.7],
            },
            "metadata": {
                "text": "Context: Source: hello world > Test > Test Test\n---\nHello World",
                "source": "hello_world.pdf",
                "Header 1": "Test",
                "Header 2": "Test Test",
            },
        }
    ]


@pytest.fixture
def sample_agent_configs_objects(sample_agent_config_dict):
    """Converts sample agent config dicts to AgentConfig Pydantic models."""
    return {name: AgentConfig(**config) for name, config in sample_agent_config_dict.items()}


@pytest.fixture
def sample_url():
    return "https://mock.uwf.edu"


@pytest.fixture
def sample_confluence_page_json(
    scope="function",
):  # Function scope allows for in-test modifications
    """Realistic single page object from Confluence API"""
    return {
        "id": "12345",
        "title": "Advising Syllabus",
        "type": "page",
        "status": "current",
        "body": {"storage": {"value": "<p>Raw HTML content...</p>"}},
        "version": {"number": 2, "when": "2023-11-27T12:05:17.897-06:00"},
        "_links": {
            "self": "https://confluence.uwf.edu/rest/api/content/12345",
            "webui": "/display/SPACE/Advising+Syllabus",
        },
    }


@pytest.fixture
def sample_space_key():
    return "test"


@pytest.fixture
def sample_ancestors(sample_parent_id):
    return [{"id": sample_parent_id, "title": "UWF Public Knowledge Base"}]


@pytest.fixture
def sample_parent_id():
    return "7641671"


# ==============================================================================
# 2. ENVIRONMENT & FILESYSTEM
#    - Environment variables (API keys) and temporary directories.
# ==============================================================================


@pytest.fixture
def valid_agents_dir(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    return agents_dir


@pytest.fixture
def valid_architectures_dir(tmp_path):
    architectures_dir = tmp_path / "architectures"
    architectures_dir.mkdir()
    return architectures_dir


@pytest.fixture
def valid_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def valid_raw_data_dir(valid_data_dir):
    raw_data_dir = valid_data_dir / "raw"
    raw_data_dir.mkdir(parents=True)
    return raw_data_dir


@pytest.fixture
def valid_processed_data_dir(valid_data_dir):
    processed_data_dir = valid_data_dir / "processed"
    processed_data_dir.mkdir(parents=True)
    return processed_data_dir


@pytest.fixture
def gemini_api_key_env(monkeypatch):
    """Sets the Gemini API key in os.environ and returns the raw string."""
    key_value = "TEST_GEMINI_API_KEY"
    monkeypatch.setenv("GEMINI_API_KEY", key_value)
    yield key_value


@pytest.fixture
def secret_gemini_api_key_env(gemini_api_key_env):
    """Returns the Gemini API key as a SecretStr object."""
    yield SecretStr(gemini_api_key_env)


@pytest.fixture
def pinecone_api_key_env(monkeypatch):
    """Sets the Pinecone API key in os.environ and returns the raw string."""
    key_value = "TEST_PINECONE_API_KEY"
    monkeypatch.setenv("PINECONE_API_KEY", key_value)
    yield key_value


# ==============================================================================
# 3. MOCKS
#    - Mock objects for external dependencies (LLMs, Databases).
# ==============================================================================


@pytest.fixture
def mock_gemini_client():
    with patch("core.execution_service.ChatGoogleGenerativeAI") as MockClient:
        yield MockClient


@pytest.fixture
def mock_gemini_dense_embedding_client():
    with patch("core.execution_service.GoogleGenerativeAIEmbeddings") as MockEmbeddingClient:
        yield MockEmbeddingClient


@pytest.fixture
def mock_pinecone_client():
    with patch("core.execution_service.Pinecone") as MockPineconeClient:
        yield MockPineconeClient


@pytest.fixture
def mock_docling_loader():
    with patch("knowledge_base.ingestion.pdf_to_markdown_converter.DoclingLoader") as MockDoclingLoader:
        yield MockDoclingLoader


@pytest.fixture
def mock_index_object():
    return MagicMock()


@pytest.fixture
def mock_vector_normalizer():
    """Patches VectorNormalizer to return data unchanged and avoid math errors."""
    with patch("knowledge_base.processing.vector_normalizer") as mock:
        # Configure normalize to just return the first argument (identity)
        mock.normalize.side_effect = lambda vectors, vector_type: "DENSE"
        yield mock

@pytest.fixture
def mock_full_KB_pipeline(valid_raw_data_dir, valid_processed_data_dir):
    """
    Creates a KnowledgeBasePipeline instance where __init__ runs normally,
    but all external dependencies (Converter, Embedders, Chunker) are mocked.
    """
    # We patch the classes inside the pipeline module so __init__ uses our Mocks
    with patch('knowledge_base.pipeline.knowledge_base_pipeline.ExecutionService') as MockExec, \
         patch('knowledge_base.pipeline.knowledge_base_pipeline.PDFToMarkdownConverter') as MockConverter, \
         patch('knowledge_base.pipeline.knowledge_base_pipeline.TextChunker') as MockChunker, \
         patch('knowledge_base.pipeline.knowledge_base_pipeline.GeminiEmbedder') as MockGemini, \
         patch('knowledge_base.pipeline.knowledge_base_pipeline.PineconeSparseEmbedder') as MockPinecone:
        
        pipeline = KnowledgeBasePipeline(
            kb_name="test_kb",
            raw_data_path=valid_raw_data_dir,
            processed_data_path=valid_processed_data_dir
        )

        pipeline.execution_service = MagicMock()
        pipeline.converter = MagicMock()
        pipeline.chunker = MagicMock()
        pipeline.gemini_embedder = MagicMock()
        pipeline.pinecone_embedder = MagicMock()
        pipeline.pc = MagicMock()

        yield pipeline

# ==============================================================================
# 4. SERVICE INSTANCES
#    - Initialized instances of your application's core services.
# ==============================================================================


@pytest.fixture
def instance_execution_service():
    """Creates an instance of ExecutionService with NO agent configs."""
    return ExecutionService()


@pytest.fixture
def instance_agent_config_execution_service(sample_agent_configs_objects):
    """Creates an instance of ExecutionService PRE-LOADED with sample agent configs."""
    return ExecutionService(agent_configs=sample_agent_configs_objects)


# ==============================================================================
# 5. KNOWLEDGE BASE INSTANCES
#    - Initialized instances of the knowledge base components.
# ==============================================================================
@pytest.fixture
def pdf_converter(valid_raw_data_dir, valid_processed_data_dir):
    """Creates an instance of PDFToMarkdownConverter with valid paths."""
    return PDFToMarkdownConverter(raw_data_path=valid_raw_data_dir, processed_data_path=valid_processed_data_dir)


@pytest.fixture
def text_chunker():
    """Returns a TextChunker instance with small chunk sizes for easier testing."""
    return TextChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def url_converter(valid_processed_data_dir, sample_url):
    """Creates an instance of URLtoMarkdownConverter with valid paths."""
    converter = URLtoMarkdownConverter(base_url=sample_url, saved_data_path=valid_processed_data_dir)
    converter.session = MagicMock()  # Replace the real requests.Session with a MagicMock
    return converter


@pytest.fixture
def confluence_processor(valid_processed_data_dir):
    processor = ConfluencePageProcessor(saved_data_path=valid_processed_data_dir)
    processor.file_saver = MagicMock()
    processor.content_extractor = MagicMock()
    return processor


@pytest.fixture
def confluence_html_extractor():
    return ConfluenceContentExtractor()
