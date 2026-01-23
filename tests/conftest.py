"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import SecretStr
from pathlib import Path
from langchain_core.documents import Document

from src.schemas.agent_schemas import AgentConfig
from src.core.execution_service import ExecutionService
from src.knowledge_base.ingestion.pdf_to_markdown_converter import PDFToMarkdownConverter
from src.knowledge_base.processing.text_chunker import TextChunker

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
            "description": "A test agent configuration following valid schema."
        },
        "model": {
            "provider": "google",
            "name": "gemini-3-pro-preview",
            "temperature": 0.5
        },
        "system_prompt": "You are a helpful assistant."
    }

@pytest.fixture
def sample_agent_config_dict():
    """Returns a dictionary containing multiple valid agent configurations."""
    return {
        'good_test_agent_1': {
            "version": "1.0",
            "agent_metadata": {
                "name": "good_test_agent_1",
                "description": "A test agent configuration following valid schema."
            },
            "model": {
                "provider": "google",
                "name": "gemini-3-pro-preview",
                "temperature": 0.5
            },
            "system_prompt": "You are a helpful assistant."
        },
        'good_test_agent_2': {
            "version": "1.0",
            "agent_metadata": {
                "name": "good_test_agent_2",
                "description": "Another test agent configuration following valid schema."
            },
            "model": {
                "provider": "google",
                "name": "gemini-3-small",
                "temperature": 0.7
            },
            "system_prompt": "You are a creative assistant."
        }
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
    text_doc = Document(
        page_content = text,
        metadata = {"Header 1": "Test Header"}
    )
    return [text_doc]
@pytest.fixture
def short_text_chunk():
    """Text chunk used to test the functionality of the TextChunker class"""
    short_text = "123456789"
    short_doc = Document(
        page_content = short_text,
        metadata = {"Header 1": "First Header", "Header 2": "Second Header "}
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
            "Header 2": "Test Test"
        }
    )
    return [text_chunk]

@pytest.fixture
def dense_embeddings():
    """
    Returns a list containing a single dense vector embedding (list of floats).
    Matches the structure expected by the upsert function (List[List[float]]).
    """
    return [[0.1, 0.2, 0.3]]

@pytest.fixture
def sparse_embeddings():
    """
    Returns a list containing a single Mock object representing a sparse embedding.
    The mock object has 'sparse_indices' and 'sparse_values' attributes.
    """
    sparse_mock = MagicMock()
    # Indices must be integers, values must be floats
    sparse_mock.sparse_indices = [1, 5]
    sparse_mock.sparse_values = [0.9, 0.8]

    # Must be returned as a list to allow zipping with chunks and dense embeddings
    return [sparse_mock]

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
    return [{
        'id': 'hello_world_chunk_1',
        'values': [0.1, 0.2, 0.3],
        'sparse_values': {
            'indices': [1,5],
            'values': [0.9, 0.8]
        },
        'metadata': {
            'text': "Context: Source: hello world > Test > Test Test\n---\nHello World",
            'source': 'hello_world.pdf',
            "Header 1": "Test",
            "Header 2": "Test Test"
        }
    }]

@pytest.fixture
def sample_agent_configs_objects(sample_agent_config_dict):
    """Converts sample agent config dicts to AgentConfig Pydantic models."""
    return {
        name: AgentConfig(**config) 
        for name, config in sample_agent_config_dict.items()
    }


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
    with patch("src.core.execution_service.ChatGoogleGenerativeAI") as MockClient:
        yield MockClient

@pytest.fixture
def mock_gemini_dense_embedding_client():
    with patch("src.core.execution_service.GoogleGenerativeAIEmbeddings") as MockEmbeddingClient:
        yield MockEmbeddingClient

@pytest.fixture
def mock_pinecone_client():
    with patch("src.core.execution_service.Pinecone") as MockPineconeClient:
        yield MockPineconeClient

@pytest.fixture
def mock_docling_loader():
    with patch("src.knowledge_base.ingestion.pdf_to_markdown_converter.DoclingLoader") as MockDoclingLoader:
        yield MockDoclingLoader

@pytest.fixture
def mock_index_object():
    return MagicMock()

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
    return PDFToMarkdownConverter(
        raw_data_path=valid_raw_data_dir,
        processed_data_path=valid_processed_data_dir
    )

@pytest.fixture
def text_chunker():
    """Returns a TextChunker instance with small chunk sizes for easier testing."""
    return TextChunker(chunk_size = 50, chunk_overlap = 10)