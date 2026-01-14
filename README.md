# AI Agents Project
Building and testing different AI agents and agentic system architectures.

## Overview 
This repository provides a lightweight framework for defining, configuring, testing, and benchmarking different agent types and multi-agent system architectures. The code emphasizes clear separation between agent implementations, configurations, and architecture coordination.

Key concepts:
- **Agent**: An individual unit 
- **Architecture**: Orchestrates one or more agents, defines communication & coordination, and may include central or distributed planning logic — implemented in `src/architectures/`.
- **Configuration**: YAMLfiles that set model parameters, tools, memory, and behavior policies for agents and architectures (in `configs/`).


## Project Structure 

* **`configs/`**:
    * `agents/`: YAML definitions for agents.
    * `architectures/`: YAML definitions for architectures.
* **`src/`**: Source code.
    * `agents/`: Agent implementations.
        * `base_agent.py`: **Active**. Single ReACT agent with tool support.
        * `coordination_agent.py`: Placeholder file for specialized agent (WIP).
        * `email_agent.py`: Placeholder file for specialized agent (WIP).
        * `rag_agent.py`: Placeholder file for specialized agent (WIP).
        * `web_search_agent.py`: Placeholder file for specialized agent (WIP).
    * `architectures/`: Logic for multi-agent orchestration (WIP).
    * `core`: Application infrastructure
        * `logging_setup.py`: Sets the logging configuration for the application.
        * `execution_service.py`: Creates LLM client based upon specified configuration file.
        * `architecture_manager.py`: (WIP)
        * `state.py`: Defines standard information included in Agent State during execution.
    * `utils`: Core utility files.
        * `application_streamer.py`: Streams the provided application graph one node at a time, streaming intermediate state. Used when executing the agent(s).
        * `architecture_diagram_generator.py`: Generates diagram of graph architecture. 
        * `config_loader.py`: Parses YAML files and validates against Pydantic models. Options available for both agent YAML files and architecture YAML files.
        * `process_events.py`: Accesses the tool calls and AI agent's responses to display to the user.
    * `knowledge_base`: RAG pipeline including PDF ingestion, chunking logic, embedding, and uploading to Vector DB (WIP).
        * `ingestion`:
            * `pdf_to_markdown_converter.py`: Uses Docling to convert PDF files to markdown.
        * `processing`:
            * `text_chunker.py`: Splits .md files on headers, then recursively chunks the headers into the specified size. 
            * `gemini_embedder.py`: (WIP).s
        * `uploading_to_vector_db`:
    * `schemas/`: Pydantic models enforcing validation rules.'
        * `agent_schemas.py`: Pydantic model for agent .YAML configuration files. 
        * `architecture_schemas.py`: Pydantic model for architecture .YAML configuration files.
    * `constants.py`: Contains constants used throughout SRC folder. Includes directory file paths.
    * `main.py`: FUTURE USE - Main entry point for the application (WIP).
* **`tests/`**: Test code.
    * `integration`: Contains integration tests.
    * `unit`: Contains unit tests (WIP).
        * `agents`: Unit tests for agent files.
        * `architecture`: Unit tests for architecture files.
        * `core`: Unit tests for core infrastructure.
        * `utils`: Unit tests for utility functions. 
        * `knowledge_base`: Unit tests for RAG pipeline.
    * `conftest.py`: Contains PyTest configurations and reusable fixtures.
* **`diagrams/`**: 
    * `production`: Contains graph diagrams for each agent configuration.

See ARCHITECTURE.md for more information on project architecture.

## Getting Started 

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the package and dev extras (recommended):
```bash
pip install -e ".[dev]"
```

3. Run a quick test to verify the environment:
```bash
# Run mypy static checks
python -m mypy src

# Run the tests
pytest tests/ -v

# Run the tests with coverage. Report shows missing lines.
pytest --cov=src --cov-report=term-missing
```

4. Setup Environment Variables

This project requires API keys to function.

First, copy the example environment file:

```bash
cp .env.example .env
```

Open .env and add the required API keys.

If you do not know the full file path to your project root, run the following command and copy the output:
```bash
# Ensure you are in the project root
pwd
```

## Running the agent

Currently, the single-agent implementation is available for testing.

```bash
# Ensure you are in the project root
python src/agents/base_agent.py
```

Note: Ensure your .env file is set up before running.

Eventually, the agentic entry point will be in the main.py file; however, the architecture is not yet finished to do so.

## Running Tests & CI 

Use pytest for local testing:

```bash
pytest tests/ -v
```

For `mypy` / type-checking, run:

```bash
python -m mypy src
```

CI should run these checks and may use a pinned `requirements-dev.txt` to ensure reproducible test environments.

## Important Notes

If downloading this package from GitHub, the data directory, which includes the raw and processed subdirectories, does not include the files uploaded to the knowledge base. 
Prior to using the ingestion, processing, or upload functions contained within src/knowledge_base, the user should upload their own files to the appropriate directory.

## Helpful Commands

To view the entire project structure from the terminal:

```bash
tree -I "__pycache__|.git|.venv|.pytest_cache"
```


 
