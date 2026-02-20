"""
LangGraph used to implement a linear RAG retriever evaluation pipeline that runs through
each question in the evaluation dataset, retrieves contexts, computes RAGAS metrics, and
saves results to a report.

The graph is designed to be modular and extensible, allowing for easy swapping of
retrieval methods, evaluation metrics, and report generation formats. It serves as the
core execution engine for the RAG evaluation process.

The graph consists of the following key nodes:
1. **Dataset Loader Node**: Loads the evaluation dataset
    (questions and ground truth answers) into the graph's state.
2. **Retrieval Node**: For each question, uses the StructuredRagRetriever to retrieve
    relevant contexts from Pinecone and stores the structured results in the state.
3. **Evaluation Node**: For each question, takes the retrieved contexts and ground truth
    answer, computes RAGAS metrics using the compute_ragas_metrics tool, and stores the results in the state.
4. **Report Generation Node**: After processing all questions, aggregates the
    per-question results into a final EvalReport and generates both JSON and Markdown reports using the ReportGenerator.
5. **Summary Node**: LLM-generated summary of the overall evaluation results,
    which can be included in the final report or used for quick analysis.

The graph is executed using the ExecutionService, which manages the flow of data between
nodes and ensures that all steps are completed in the correct order.

The final output includes a detailed report of the RAG evaluation results, which can be
used to analyze retriever performance and identify areas for improvement.
"""

import logging
from typing import List, Union
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from ragas.llms import InstructorBaseRagasLLM
from langgraph.graph import StateGraph, END
from rag_eval.schemas.eval_schemas import (
    EvalAgentState,
    QuestionEvalResult,
    EvalReport,
)
from rag_eval.components.structured_rag_retriever import StructuredRagRetriever
from rag_eval.evaluation_dataset_loader import EvaluationDatasetLoader
from rag_eval.components.ragas_metrics import compute_ragas_metrics
from rag_eval.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

load_dotenv()

# Define nodes for the evaluation graph


def build_eval_graph(
    csv_dir: Union[str, Path],
    csv_filename: str,
    rag_retriever: StructuredRagRetriever,
    top_k_matches: int,
    eval_llm: InstructorBaseRagasLLM,
    summary_llm: BaseChatModel,
    report_generator: ReportGenerator,
    dataset_name: str,
    dataset_description: str,
    encoding: str = "utf-8",

):
    def load_dataset(state: EvalAgentState) -> dict:
        """
        Loads the evaluation dataset from a specified CSV file using the
        EvaluationDatasetLoader component.

        Returns the loaded dataset as a list of EvalDatasetRow objects to be stored in
        the graph's state under the 'loaded_dataset' key.
        """
        logger.info("Utilizing load_dataset node to load evaluation dataset into state.")
        loader = EvaluationDatasetLoader(csv_dir=csv_dir, encoding=encoding)
        # Returns a list of EvalDatasetRow objects
        rows = loader.load_eval_dataset(csv_filename=csv_filename)
        return {"loaded_dataset": rows}

    def run_retrieval(state: EvalAgentState) -> dict:
        """
        Performs retrieval for each question in the loaded dataset using the provided
        StructuredRagRetriever.

        Stores the structured retrieval results in the graph's state under the
        'retrieval_results' key

        Skips any questions that encounter retrieval errors,
        logging the issue and continuing with the next question to ensure robustness
        of the evaluation process.

        Raises:
            RuntimeError if no successful retrievals are made.

        Returns:
            A dictionary containing a list of RetrievalResult objects for each question.
            Dict will be stored in state under 'retrieval_results' key for use in
            subsequent metric computation.

        """
        logger.info(
            """Utilizing run_retrieval node to perform RAG similarity search 
                    for each question in the dataset."""
        )
        retrieval_results = []
        logger.info(
            f"Running retrieval for {len(state['loaded_dataset'])} questions with top_k_matches={top_k_matches}"
        )
        # Iterate over list of EvalDatasetRow objects
        for row in state["loaded_dataset"]:
            question = row.question
            try:
                result = rag_retriever.retrieve(user_query=question,
                                                top_k_matches=top_k_matches)
                retrieval_results.append(result)
            except Exception as e:
                logger.error(f"Error during retrieval for question '{question[:50]}...': {e}")
                # Skip this question
                continue
        
        if len(retrieval_results) == 0:
            value_error_msg = """No successful retrievals.
            Check for issues with the retriever or dataset."""
            logger.error(value_error_msg)
            raise RuntimeError(value_error_msg)
        else:
            return {"retrieval_results": retrieval_results}

    async def compute_metrics(state: EvalAgentState) -> dict:
        """
        Computes RAGAS metrics for each question using the compute_ragas_metrics tool.
        
        Tool takes the question, retrieved contexts, and ground truth answer as input
        and returns a QuestionEvalResult containing the precision and recall scores.

        Async function because RAGAS metric computation involves LLM calls.
        """
        logger.info("""Utilizing compute_metrics node to compute RAGAS metrics
                    for each question.""")

        # O(n) time complexity for dict creation.
        # Eval datasets are typically 50-100 questions. Should be maintainable for now.
        # TODO: For larger datasets, evaluate adding primary key and creating a dict for O(1) lookups of ground truth answers"""

        ground_truth_map = {
            row.question: row.ground_truth for row in state["loaded_dataset"]
        }  # Create a mapping of question to ground truth for easy lookup

        ragas_results = []
        for result in state["retrieval_results"]:  # Iterate over list of RetrievalResult objects
            question = result.question
            retrieved_contexts = result.contexts

            # Map question to corresponding ground truth answer. O(1) lookup with dict.
            ground_truth = ground_truth_map.get(question, "")
            if not ground_truth:
                logger.warning(
                    f"""No ground truth answer found for question '{question[:50]}...'.
                    Skipping metric computation for this question."""
                )
                continue

            try:
                eval_result = await compute_ragas_metrics(
                    evaluator_llm=eval_llm,
                    question=question,
                    retrieved_contexts=retrieved_contexts,
                    ground_truth=ground_truth,
                )
                ragas_results.append(eval_result)
            except Exception as e:
                logger.error(
                    f"""Error computing RAGAS metrics for question
                    '{question[:50]}...': {e}""")
                continue
        
        if len(ragas_results) == 0:
            value_error_msg = """No successful RAGAS metric computations.
            Check for issues with the evaluation LLM, retriever results, or dataset."""
            logger.error(value_error_msg)
            raise RuntimeError(value_error_msg)
        else:
            logger.info(f"""Successfully computed RAGAS metrics for {len(ragas_results)}
                        questions.""")
            # Return list of QuestionEvalResult objects
            # Stored in state under 'eval_results' key for use in report generation
            return {"eval_results": ragas_results}
    def generate_eval_report(state: EvalAgentState) -> dict:
        """
        Aggregates per-question RAGAS results into a final EvalReport and generates
        JSON and Markdown reports using the ReportGenerator component.

        The generated report file paths are returned in the state for reference.
        """
        logger.info("""Utilizing generate_report node to aggregate RAGAS results into a
                    final report and generate JSON and Markdown outputs.""")

        # Aggregate per-question results into final report
        results_list = state["eval_results"]  # List of QuestionEvalResult objects

        # Using np arrays for efficient aggregation of precision and recall scores.
        recall_scores, precision_scores = zip(*[(r.context_recall, r.context_precision) for r in results_list])
        average_recall = float(np.mean(recall_scores))
        average_precision = float(np.mean(precision_scores))

        report = EvalReport(
            average_context_recall=average_recall,
            average_context_precision=average_precision,
            total_questions_evaluated=len(results_list),
            dataset_name=dataset_name,
            per_question_results=results_list,
            description=dataset_description,
        )
        json_path, md_path = report_generator.generate_report(report=report)
        return {"final_report": report,
                "json_report_path": str(json_path),
                "md_report_path": str(md_path)}
    def summary_agent(state: EvalAgentState) -> dict: #type: ignore
        """
        Generates a brief summary of the evaluation results using an LLM.

        This summary can be included in the final report or used for quick analysis
        of retriever performance.

        Returns:
            A dictionary containing the generated summary string under the 'summary' key.
        """
        logger.info("""Utilizing summary_agent node to generate a brief summary of the
                    evaluation results using an LLM.""")

        report = state["final_report"]
        summary_prompt = SystemMessage(content = f"""Summarize the following RAG 
        evaluation results in 2-3 sentences, highlighting key insights about
        retriever performance and areas for improvement:
        
        Dataset Name: {report.dataset_name}
        Dataset Description: {report.description}
        Total Questions Evaluated: {report.total_questions_evaluated}
        
        Average Context Precision: {report.average_context_precision:.2f}
        Average Context Recall: {report.average_context_recall:.2f}
        
        JSON Report Path: {state['json_report_path']}
        MD Report Path: {state['md_report_path']}

        Then, provide a more-detailed analysis of the per-question results, 
        identifying any patterns in performance or issues with retrieval quality.
        
        Per-Question Results:
        {[(r.question, r.context_recall, r.context_precision) for r in report.per_question_results]}
        
        Provide a concise analysis of these results.""")

        messages = [summary_prompt] + list(state["messages"])
        max_retries = 3

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} to generate summary with LLM.")
                response = summary_llm.invoke(messages)

                # Update state with the new messages (user message + AI response)
                return {"llm_summary": response.content}
            except Exception as e:
                logger.error(f"""Error generating summary with LLM on
                             attempt {attempt + 1}: {e}.""")
                if attempt < max_retries - 1:
                    logger.info("Retrying summary generation...")
                else:
                    logger.error("Max retries reached. Failed to generate summary.")
                    return {"llm_summary": ""}  # Return empty summary to avoid blocking the graph

        
    graph = StateGraph(EvalAgentState)
    graph.add_node("load_dataset", load_dataset)
    graph.add_node("run_retrieval", run_retrieval)
    graph.add_node("compute_metrics", compute_metrics)
    graph.add_node("generate_eval_report", generate_eval_report)
    graph.add_node("summary_agent", summary_agent)
    graph.set_entry_point("load_dataset")
    graph.add_edge("load_dataset", "run_retrieval")
    graph.add_edge("run_retrieval", "compute_metrics")
    graph.add_edge("compute_metrics", "generate_eval_report")
    graph.add_edge("generate_eval_report", "summary_agent")
    graph.add_edge("summary_agent", END)
    return graph.compile()
