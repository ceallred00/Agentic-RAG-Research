import logging
from typing import List
from ragas.metrics.collections import ContextRecall, ContextPrecision
from ragas.llms import InstructorBaseRagasLLM
from ragas.exceptions import RagasException
from rag_eval.schemas.eval_schemas import QuestionEvalResult

logger = logging.getLogger(__name__)

async def compute_ragas_metrics(
        evaluator_llm: InstructorBaseRagasLLM,
        question: str,
        retrieved_contexts: List[str],
        ground_truth: str) -> QuestionEvalResult:
    """
    Computes Context Precision and Context Recall for a single evaluation question
    using the RAGAS v0.4 collections API.

    Args:
        evaluator_llm (InstructorBaseRagasLLM): A RAGAS LLM instance (created via llm_factory)
                                                 used internally by the metrics to judge relevance.
        question (str): The natural language query being evaluated.
        retrieved_contexts (List[str]): The text content of each chunk retrieved from Pinecone.
        ground_truth (str): The expected correct answer, used by RAGAS as the reference
                            to judge retrieval quality.

    Returns:
        QuestionEvalResult: Contains the question, context_precision score,
                            context_recall score, and the retrieved contexts.

    Raises:
        RagasException: If RAGAS fails to parse the LLM response or the LLM does not finish.
        Exception: Any unexpected error during metric computation.
    """
    logger.info(f"Computing RAGAS metrics for {question[:50]}")
    try:
        logger.info("Computing precision score")
        precision_scorer = ContextPrecision(llm=evaluator_llm)
        # Returns a MetricResult object
        precision_results = await precision_scorer.ascore(
            user_input = question,
            retrieved_contexts= retrieved_contexts,
            reference = ground_truth
        )
        context_precision_score = precision_results.value

        logger.info("Computing recall score")
        recall_scorer = ContextRecall(llm=evaluator_llm)
        # Returns a MetricResult object
        recall_results = await recall_scorer.ascore(
            user_input = question,
            retrieved_contexts = retrieved_contexts,
            reference = ground_truth
        )
        context_recall_score = recall_results.value
    except RagasException as e:
        logger.error(f"RAGAS error for question '{question[:50]}...': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error computing RAGAS metrics for question '{question[:50]}...':{e}")
        raise

    return QuestionEvalResult(
        question = question,
        context_precision= context_precision_score,
        context_recall = context_recall_score,
        contexts = retrieved_contexts
    )
