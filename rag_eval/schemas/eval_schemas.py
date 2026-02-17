"""Defines Pydantic models for the evaluation agent and the dataset."""


from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict, Sequence, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# TODO: Evaluate adding expected contexts (Header Hierachy, URL, etc.)

class EvalDatasetRow(BaseModel):
    """
    Represents a single evaluation sample with a question and its ground truth answer
    """
    question: Annotated[str, Field(min_length=1,description = "Natural language query to evaluate retrieval against.")]
    ground_truth: Annotated[str, Field(min_length=1, description = "Expected correct answer used by RAGAS to judge context recall.")]

class RetrievalResult(BaseModel):
    """
    Aggregates all Pinecone matches for a single evaluation query, including retrieved text, scores, and
    metadata
    """
    question: Annotated[str, Field(description = "Natural language query to evaluate retrieval against.")]
    contexts: Annotated[List[str], Field(description = "Per-match enriched contexts used by Pinecone to evaluate similarity score.")]
    metadata: Annotated[List[dict], Field(description = "Per-match metadata dicts containing source, headers, URL, and other fields stored during ingestion.")]
    scores: Annotated[List[float], Field(description = "List of similarity score returned by Pinecone for each match. Higher scores indicate higher similarity to the user's query")]
    ids: Annotated[List[str], Field(description = "List of vector IDs returned for traceability.")]

class QuestionEvalResult(BaseModel):
    """
    Represents the returned RAGAS metrics for a single evaluation query, as well as the retrieved contexts.
    """
    question: Annotated[str, Field(description = "Natural language query to evaluate retrieval against.")]
    context_precision: Annotated[float, Field(description = "Metric that evaluates the retriever's ability to rank relevant chunks higher than irrelevant chunks for a given query. The returned value should be between 0 and 1.")]
    context_recall: Annotated[float, Field(description = "Measures how many of the relevant pieces of information were successfully retrieved.")]
    contexts: Annotated[List[str], Field(description = "Per-match enriched contexts used by Pinecone to evaluate similarity score.")]

class EvalReport(BaseModel):
    """ Final aggregation of information."""
    average_context_recall: Annotated[float, Field(description = "Average context recall aggregated over the entire dataset")]
    average_context_precision: Annotated[float, Field(description = "Average context precision aggregated over the entire dataset")]
    total_questions_evaluated: Annotated[int, Field(description = "Number of questions evaluated.")]
    dataset_name: Annotated[str, Field(description = "Name of evaluated dataset.")]
    per_question_results: Annotated[List[QuestionEvalResult], Field(description = "Per-question results, including the question, context and precision scores, and the retrieved contexts.")]


class EvalAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Accumulated messages with add_messages reducer
    loaded_dataset: List[EvalDatasetRow]  # Natural language queries and associated ground truth responses
    retrieval_results: List[RetrievalResult]  # Aggregated Pinecone matches for every question in the dataset
    eval_results: List[QuestionEvalResult]  # RAGAS metrics and retrieved contexts for each question
    final_report: EvalReport  # Aggregated performance report across the full dataset



