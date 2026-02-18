# System Architecture

This document describes the key architectural decisions and data flows in the AI Agents project.

---

## 1. Configuration Flow

The system uses a validate-on-load strategy to ensure type safety before any execution begins.

| Step | Component | Responsibility | Decision Logic |
| :--- | :--- | :--- | :--- |
| **Load** | `ConfigLoader` | Reads raw YAML files from the specified subdirectory. | **Skip & Log:** Malformed YAML files are skipped to prevent a full crash. |
| **Validate** | `Pydantic Models` | Checks data integrity against schemas immediately after loading. | **Skip & Log:** Files containing invalid schemas are skipped; bad data never enters the system. |
| **Execute** | `ExecutionService` | Receives and processes only valid configuration objects. | **Raise Error:** If the requested agent config is not found among the valid configs, execution stops. |

---

## 2. Agent Architecture

The active agent (`base_agent.py`) is a **ReACT (Reasoning + Acting)** agent implemented with LangGraph.

**Execution flow:**
1. `ConfigLoader` loads and validates the agent's YAML configuration from `configs/agents/`.
2. `ExecutionService` creates the appropriate LLM client (Eden AI proxy for the base agent).
3. The agent graph is compiled with the agent node and a tool node.
4. `ApplicationStreamer` runs the graph, streaming one node update at a time.
5. `ProcessEvents` displays tool calls and model responses to the user as they arrive.
6. LangGraph's `MemorySaver` checkpoint maintains conversation history across turns within a session.

**Tools available to the base agent:**

| Tool | Description |
| :--- | :--- |
| `perform_rag_search` | Hybrid vector search over the Pinecone knowledge base. |
| `search_web` | Web search for information not in the knowledge base. |
| `search_for_advisor` | Looks up the assigned academic advisor for a student. |
| `draft_email` | Drafts an email to the student's advisor. |
| `send_email` | Sends the drafted email. |
| `end_conversation` | Gracefully ends the conversation session. |

---

## 3. Knowledge Base Pipeline

The RAG pipeline transforms raw documents into searchable hybrid vectors in Pinecone. All steps are orchestrated by `knowledge_base_pipeline.py`.

```
Raw Documents (PDF / Markdown)
        │
        ▼
 [Ingestion Layer]
  PDFToMarkdownConverter  ── Docling parses PDFs into Markdown
  ConfluenceContentExtractor ── BeautifulSoup parses Confluence HTML → Markdown
  FileSaver               ── Saves processed Markdown to data/processed/
        │
        ▼
 [Processing Layer]
  TextChunker             ── Header-based split → recursive character split
                             (2000 char chunks, 400 char overlap)
        │
        ├──▶ GeminiEmbedder        ── Dense vectors (768-dim, Gemini embedding-001)
        └──▶ PineconeSparseEmbedder ── Sparse vectors (sparse-english-v0)
        │
        ▼
  VectorNormalizer        ── L2 normalization applied to both vector types
        │
        ▼
 [Vector DB Layer]
  UpsertToVectorDB        ── Batched hybrid upsert to Pinecone (50 records/batch)
```

**Key design decisions:**

- **Hybrid search (dense + sparse)**: Dense vectors capture semantic meaning; sparse vectors capture keyword overlap. Together they improve both recall and precision.
- **Header-first chunking**: The `TextChunker` splits on Markdown headers before falling back to character limits, preserving document structure in each chunk.
- **Retry with exponential backoff**: All embedding API calls use `retry.py` to handle transient rate-limit errors gracefully without crashing the pipeline.
- **Batch limits**: Gemini batches up to 100 texts; Pinecone upserts up to 50 records per request, staying under the 2 MB request size limit.

---

## 4. RAG Retrieval (Query Time)

When the agent calls `perform_rag_search`, the following happens:

1. The query string is embedded with both `GeminiEmbedder` (dense, RETRIEVAL_QUERY task type) and `PineconeSparseEmbedder` (sparse).
2. Both vectors are L2-normalized.
3. A hybrid query is sent to Pinecone, retrieving the top-5 matches.
4. Results are returned as formatted text with source metadata and relevance scores.

---

## 5. RAG Evaluation Module

The `rag_eval/` module benchmarks retrieval quality using RAGAS metrics.

**Data flow:**

```
CSV Dataset (question, ground_truth)
        │
        ▼
 EvaluationDatasetLoader  ── File validation + row-level Pydantic validation
        │
        ▼
 [Evaluation Agent]        ── Retrieves contexts for each question via RAG tool
        │
        ▼
 [RAGAS Metrics]
  context_precision        ── Are retrieved chunks relevant to the question?
  context_recall           ── Do retrieved chunks cover the ground truth answer?
        │
        ▼
 ReportGenerator           ── Writes timestamped JSON + Markdown reports
                               to rag_eval/results/
```

**Schema summary (`rag_eval/schemas/eval_schemas.py`):**

| Model | Purpose |
| :--- | :--- |
| `EvalDatasetRow` | One row from the evaluation CSV (question + ground_truth). |
| `RetrievalResult` | Aggregated Pinecone matches for a single question (contexts, metadata, scores). |
| `QuestionEvalResult` | RAGAS metric scores + retrieved contexts for one question. |
| `EvalReport` | Final report aggregating average metrics and all per-question results. |
| `EvalAgentState` | LangGraph TypedDict for the evaluation agent's state. |
