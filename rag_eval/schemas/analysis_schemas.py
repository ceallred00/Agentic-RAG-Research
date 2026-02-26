from pydantic import BaseModel, Field
from typing import Annotated, List

class AggregatedPoorResult(BaseModel):
    question: Annotated[str, Field(description = "Natural language query to evaluate retrieval against.")]
    scores: Annotated[List[float], Field(description = "List of metrics beneath the pre-defined threshold")]
    avg_score: Annotated[float, Field(description = "Average metric across runs")]
    runs_below_threshold: Annotated[int, Field(description = "Number of runs which had metrics below the specified threshold.")]
    contexts_per_run: Annotated[List[List[str]], Field(description = "Contexts retrieved per failing run, for diagnosing retrieval consistency.")]

class EvalReportSummary(BaseModel):
    file_names: Annotated[List[str],Field(description = "List of evaluated report filenames from which the analysis was generated.")]
    dataset_name: Annotated[str, Field(description = "Name of evaluated dataset.")]
    description: Annotated[str, Field(description = "Brief description of the evaluation run (e.g., 'Baseline hybrid search, top_k=5')")]
    cross_run_average_context_recall: Annotated[float, Field(description = "Mean context recall score averaged across all evaluated pipeline runs.")]
    cross_run_average_context_precision: Annotated[float, Field(description = "Mean context precision score averaged across all evaluated pipeline runs.")]
    cross_run_std_context_recall: Annotated[float, Field(description = "Sample standard deviation of context recall scores across pipeline runs. Indicates score consistency.")]
    cross_run_std_context_precision: Annotated[float, Field(description = "Sample standard deviation of context precision scores across pipeline runs. Indicates score consistency.")]
    per_run_context_recall: Annotated[List[float], Field(description = "Context recall score for each individual pipeline run, in the same order as file_names.")]
    per_run_context_precision: Annotated[List[float], Field(description = "Context precision score for each individual pipeline run, in the same order as file_names.")]
    poor_recall_results: Annotated[List[AggregatedPoorResult], Field(description = "List of AggregatedPoorResult objects below the threshold")]
    poor_precision_results: Annotated[List[AggregatedPoorResult], Field(description = "List of AggregatedPoorResult objects below the threshold")]

