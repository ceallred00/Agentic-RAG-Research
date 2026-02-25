# AI Agents Project
Building and testing different AI agents and agentic system architectures.

## Overview
This repository provides a framework for defining, configuring, testing, and benchmarking AI agents and multi-agent system architectures. The primary application is a UWF student assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from a university knowledge base.

Key concepts:
- **Agent**: An individual unit that reasons and acts using tools.
- **Architecture**: Orchestrates one or more agents, defines communication & coordination, and may include central or distributed planning logic — implemented in `src/architectures/`.
- **Configuration**: YAML files that set model parameters, tools, memory, and behavior policies for agents and architectures (in `configs/`).
- **Knowledge Base**: A RAG pipeline that ingests documents, chunks and embeds them, and uploads vectors to Pinecone for hybrid (dense + sparse) retrieval.
- **RAG Evaluation**: A module for benchmarking retrieval quality using RAGAS metrics.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deeper look at design decisions.

---

## Project Structure

```
.
├── configs/
│   ├── agents/                     # YAML definitions for individual agents
│   │   ├── base_agent.yaml         # Active ReACT agent configuration
│   │   ├── rag_agent.yaml
│   │   ├── coordination_agent.yaml
│   │   ├── email_agent.yaml
│   │   └── web_search_agent.yaml
│   ├── eval/                       # YAML configuration for the RAG evaluation pipeline
│   │   └── eval_config.yaml        # Specifies RAGAS LLM, summary LLM, retriever, report, and data settings.
│   └── architectures/              # YAML definitions for multi-agent architectures (WIP)
│
├── src/
│   ├── agents/                     # Agent implementations
│   │   ├── base_agent.py           # Active. ReACT agent with 6 tools (RAG, web search, email, advisor lookup).
│   │   ├── rag_agent.py            # Placeholder — WIP
│   │   ├── coordination_agent.py   # Placeholder — WIP
│   │   ├── web_search_agent.py     # Placeholder — WIP
│   │   └── email_agent.py          # Placeholder — WIP
│   │
│   ├── architectures/              # Multi-agent orchestration logic (WIP)
│   │
│   ├── core/                       # Application infrastructure
│   │   ├── execution_service.py    # Factory for LLM (Gemini, Eden AI), embedding, and Pinecone clients.
│   │   ├── agent_state.py          # Defines AgentState TypedDict with message history.
│   │   ├── logging_setup.py        # Dual logging: file handler (INFO) + configurable console handler (INFO default).
│   │   └── architecture_manager.py # Orchestrates multi-agent architectures (WIP).
│   │
│   ├── tools/                      # LangGraph tool implementations
│   │   ├── perform_rag_tool.py     # Hybrid RAG search (dense + sparse) over the Pinecone knowledge base.
│   │   └── rag_retriever.py        # Shared RagRetriever class used by both the agent and eval pipeline.
│   │
│   ├── knowledge_base/             # End-to-end RAG ingestion pipeline
│   │   ├── ingestion/
│   │   │   ├── pdf_to_markdown_converter.py    # Converts PDFs to Markdown using Docling.
│   │   │   ├── confluence_content_extractor.py # Parses Confluence HTML → Markdown via BeautifulSoup.
│   │   │   ├── file_saver.py                   # Saves processed markdown files to disk.
│   │   │   ├── confluence_page_processor.py    # Transforms raw Confluence API page data → Markdown with YAML frontmatter.
│   │   │   └── url_to_md_converter.py          # Recursively scrapes a Confluence page tree via REST API (handles pagination).
│   │   ├── processing/
│   │   │   ├── text_chunker.py             # Hierarchical markdown chunking (headers → recursive char split).
│   │   │   ├── gemini_embedder.py          # Dense embeddings via Gemini embedding-001 (768-dim).
│   │   │   ├── pinecone_sparse_embedder.py # Sparse embeddings via Pinecone sparse-english-v0.
│   │   │   ├── vector_normalizer.py        # L2 normalization for dense and sparse vectors.
│   │   │   └── retry.py                    # Exponential backoff retry utility for rate-limited APIs.
│   │   ├── vector_db/
│   │   │   ├── create_vector_db_index.py   # Creates Pinecone index (768-dim, dotproduct, serverless).
│   │   │   └── upsert_to_vector_db.py      # Batched upsert of hybrid-embedded chunks into Pinecone.
│   │   └── pipeline/
│   │       └── knowledge_base_pipeline.py  # Orchestrates the full ingestion flow end-to-end.
│   │
│   ├── schemas/                    # Pydantic models for configuration validation
│   │   ├── agent_schemas.py        # Validates agent YAML files (provider, model, temperature, etc.).
│   │   └── architecture_schemas.py # Validates architecture YAML files (WIP).
│   │
│   ├── utils/
│   │   ├── config_loader.py                    # Loads and validates YAML configs against Pydantic schemas.
│   │   ├── application_streamer.py             # Streams agent graph execution node-by-node.
│   │   ├── process_events.py                   # Displays tool calls and AI responses from streamed events.
│   │   └── architecture_diagram_generator.py   # Generates visual diagrams of agent graph architectures.
│   │
│   ├── constants.py                # Global constants: directory paths, embedding limits, batch sizes.
│   └── main.py                     # Future main entry point (WIP).
│
├── rag_eval/                       # RAG evaluation module
│   ├── run_eval.py                   # CLI entry point for running the evaluation pipeline.
│   ├── run_dataset_generator.py      # CLI entry point for generating evaluation datasets.
│   ├── eval_graph.py                 # LangGraph evaluation pipeline (load → retrieve → score → report → summarize).
│   ├── dataset_generator.py          # DatasetGenerator class: samples KB docs and generates Q&A pairs via LLM.
│   ├── evaluation_dataset_loader.py  # Loads and validates evaluation datasets from CSV files.
│   ├── report_generator.py           # Generates timestamped JSON and Markdown evaluation reports.
│   ├── components/
│   │   ├── ragas_metrics.py          # Async RAGAS metric computation (ContextPrecision, ContextRecall).
│   │   └── structured_rag_retriever.py  # Adapter wrapping RagRetriever to return RetrievalResult objects.
│   ├── schemas/
│   │   ├── eval_schemas.py           # Pydantic models: EvalDatasetRow, RetrievalResult, QuestionEvalResult, EvalReport, EvalAgentState, LLM/retriever configs.
│   │   └── dataset_schemas.py        # Pydantic models: QAPair, QAPairList (LLM output), DatasetRow (CSV row).
│   ├── datasets/                     # Evaluation dataset CSV files (not tracked in git).
│   └── results/                      # Evaluation output reports (not tracked in git).
│
├── tests/
│   ├── conftest.py                 # Pytest configuration and shared fixtures.
│   ├── unit/
│   │   ├── agents/
│   │   ├── core/
│   │   ├── tools/
│   │   ├── utils/
│   │   ├── rag_eval/
│   │   └── knowledge_base/
│   │       ├── ingestion/
│   │       ├── processing/
│   │       ├── vector_db/
│   │       └── pipeline/
│   └── integration/
│       └── knowledge_base/
│
├── diagrams/
│   ├── production/                 # Exported graph diagrams for active agent configurations.
│   └── research/                   # Diagrams from experimental/research work.
│
├── research/
│   └── LangGraph/                  # LangGraph experiments and prototypes.
│
├── data/                           # Not tracked in git — see Important Notes below.
│   ├── raw/                        # Raw input documents.
│   └── processed/                  # Processed markdown files ready for chunking.
│
├── logs/                           # Application log files (auto-created at runtime).
├── Makefile                        # Common development commands.
├── pyproject.toml                  # Project metadata and dependencies.
├── .env.example                    # Template for required environment variables.
└── .pre-commit-config.yaml         # Pre-commit hook configuration.
```

---

## Getting Started

### 1. Clone and create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install the package

```bash
# Install with dev dependencies (recommended)
pip install -e ".[dev]"

# Or via Makefile
make install-dev
```

### 3. Configure environment variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

Open `.env` and set the following:

| Variable | Description |
| :--- | :--- |
| `GEMINI_API_KEY` | Required for generating dense query and document embeddings. |
| `EDEN_AI_API_KEY` | Required to run the agent (proxied through OpenAI interface). |
| `PINECONE_API_KEY` | Required for RAG retrieval and sparse embedding generation. |
| `PROJECT_ROOT` | Absolute path to the project root directory. |

To get the absolute path to your project root:

```bash
pwd
```

### 4. Verify the setup

```bash
# Run static type checks
python -m mypy src

# Run the full test suite
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Running the Agent

The base ReACT agent is the active single-agent implementation:

```bash
python src/agents/base_agent.py
```

The agent supports 6 tools: RAG search over the UWF knowledge base, web search, advisor lookup, email drafting, email sending, and conversation end.

> Note: Ensure your `.env` file is configured before running. A Pinecone index with the knowledge base vectors must also be populated (see [Building the Knowledge Base](#building-the-knowledge-base) below).

---

## Building the Knowledge Base

The knowledge base pipeline ingests documents, chunks them, generates hybrid embeddings, and uploads them to Pinecone.

1. Place raw PDF or Markdown files in `data/raw/`.
2. Run the pipeline:

```bash
python src/knowledge_base/pipeline/knowledge_base_pipeline.py
```

Processed markdown files are saved to `data/processed/` and vectors are upserted to your Pinecone index in batches of 50.

---

## Generating an Evaluation Dataset

The dataset generator samples documents from the knowledge base and uses an LLM to produce realistic Q&A pairs for RAG retrieval evaluation.

```bash
.venv/bin/python -m rag_eval.run_dataset_generator \
    --sample-size 50 \
    --output-filename my_dataset.csv
```

Key options:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--sample-size` | *(required)* | Number of KB documents to randomly sample. |
| `--output-filename` | *(required)* | Name of the output CSV file. |
| `--n-questions` | `1` | Number of Q&A pairs to generate per document. |
| `--model` | `anthropic/claude-haiku-4-5` | Eden AI model in `provider/model` format. |
| `--temperature` | `0.0` | LLM sampling temperature. `0.0` = deterministic. |
| `--output-dir` | `rag_eval/datasets` | Directory to save the generated CSV. |
| `--min-doc-length` | `200` | Minimum document character length to include. |

The current date is automatically prepended to the filename in `YYYYMMDD_` format — passing `--output-filename dataset_batch_1.csv` produces `20260225_dataset_batch_1.csv`. Generated CSV files are saved to `rag_eval/datasets/`.

---

## Running the RAG Evaluation

The evaluation pipeline retrieves contexts from Pinecone for each question in a dataset, computes RAGAS metrics (Context Precision and Context Recall), generates a report, and produces an LLM summary.

```bash
.venv/bin/python -m rag_eval.run_eval \
    --csv-filename 20260225_dataset_batch_1.csv \
    --dataset-name "KB Baseline Batch 1" \
    --dataset-description "50-question batch, hybrid search, top_k=5"
```

Key options:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--csv-filename` | *(required)* | CSV file in `rag_eval/datasets/`. |
| `--dataset-name` | *(required)* | Name label used in the report. |
| `--dataset-description` | *(required)* | Description of the run for reporting context. |
| `--config-path` | `configs/eval/eval_config.yaml` | Path to the evaluation configuration YAML. |

Reports (JSON + Markdown) are saved to `rag_eval/results/`. The LLM summary is printed to stdout.

> Note: A populated Pinecone index is required. See [Building the Knowledge Base](#building-the-knowledge-base).

---

## Running Tests & CI

```bash
# Run all tests
pytest tests/ -v
make test

# Run with HTML coverage report
pytest tests/ --cov=src --cov-report=html
make test-cov

# Type checking
python -m mypy src

# Linting
make lint

# Auto-format code
make format
```

---

## Makefile Reference

| Command | Description |
| :--- | :--- |
| `make install` | Install project dependencies. |
| `make install-dev` | Install with dev dependencies. |
| `make test` | Run the test suite. |
| `make test-cov` | Run tests with an HTML coverage report. |
| `make lint` | Run pylint on `src/` and `tests/`. |
| `make format` | Auto-format code with Black. |
| `make clean` | Remove `__pycache__`, `.pytest_cache`, build artifacts. |

---

## Important Notes

- The `data/` directory (raw and processed files) is not tracked in git. Prior to running the ingestion or pipeline scripts, add your own source documents to `data/raw/`.
- The `rag_eval/datasets/` and `rag_eval/results/` directories are also not tracked in git.
- The `logs/` directory is created automatically at runtime.

---

## Helpful Commands

```bash
# View the project tree (excludes build/cache directories)
tree -I "__pycache__|.git|.venv|.pytest_cache"

# List all available Makefile commands
make help
```
