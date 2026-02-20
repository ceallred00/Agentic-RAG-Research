"""
CLI entry point for the RAG evaluation pipeline.

Loads the evaluation configuration from a YAML file, wires up all dependencies
(retriever, LLMs, report generator), and runs the evaluation graph against a
specified dataset.

Usage:
    python -m rag_eval.run_eval \\
        --csv-filename KB_testing_dataset.csv \\
        --dataset-name "KB Testing Baseline" \\
        --dataset-description "Baseline hybrid search evaluation, top_k=5"

** Make sure you are in your virtual environment with all dependencies installed 
before running the above command. **

Optional arguments:
    --config-path   Path to the evaluation config YAML.
                    Defaults to configs/eval/eval_config.yaml.

Output:
    - JSON and Markdown reports saved to the output directory specified in the config.
    - LLM-generated summary printed to stdout.
"""

import argparse
import asyncio
import logging

import yaml
from ragas.llms import llm_factory

from rag_eval.schemas.eval_schemas import EvalAgentConfig
from rag_eval.components.structured_rag_retriever import StructuredRagRetriever
from rag_eval.eval_graph import build_eval_graph
from rag_eval.report_generator import ReportGenerator
from core.execution_service import ExecutionService
from tools.rag_retriever import RagRetriever
from knowledge_base.processing.gemini_embedder import GeminiEmbedder
from knowledge_base.processing.pinecone_sparse_embedder import PineconeSparseEmbedder

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run the RAG evaluation pipeline.")
parser.add_argument(
    "--csv-filename",
    required=True,
    type=str,
    help="Name of the CSV dataset file located in 'rag_eval/datasets' folder.",
)
parser.add_argument(
    "--dataset-name", required=True, type=str, help="Name of the evaluation dataset. Used for reporting purposes."
)
parser.add_argument(
    "--dataset-description",
    required=True,
    type=str,
    help="Brief description of the evaluation dataset and its characteristics. Used for reporting purposes.",
)
parser.add_argument(
    "--config-path", default="configs/eval/eval_config.yaml", help="Path to the evaluation configuration YAML file."
)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    # Validate the loaded configuration against the EvalAgentConfig Pydantic model
    validated_config = EvalAgentConfig(**raw_config)

    # Not passing the validated_config to ExecutionService in this implementation
    # since the current ExecutionService implementation is designed for agent execution
    # and not specifically for the evaluation agent.
    execution_service = ExecutionService()

    pc_client = execution_service.get_pinecone_client()
    async_client = execution_service.get_eden_ai_async_client()
    dense_embedder = GeminiEmbedder(execution_service=execution_service)
    sparse_embedder = PineconeSparseEmbedder(execution_service=execution_service, pinecone_client=pc_client)

    rag_retriever = RagRetriever(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        pc_client=pc_client,
        index_name=validated_config.retriever.index_name,
    )
    structured_retriever = StructuredRagRetriever(rag_retriever=rag_retriever)

    # Configure RAGAS Evaluation LLM
    ragas_llm = llm_factory(
        model=validated_config.ragas_llm_model.model_name,
        provider=validated_config.ragas_llm_model.provider,
        client=async_client,
    )

    # Configure Summary LLM
    # Returns configured ChatOpenAI client instance based on model name provided in the configuration.
    # ChatOpenAI is used as EdenAI proxy.
    summary_llm = execution_service.get_eden_ai_client(model_name=validated_config.summary_llm_model.model_name)

    # Configure Report Generator Instance
    eval_report_generator = ReportGenerator(output_dir=validated_config.report.output_dir)

    # Build the evaluation graph with the configured components and validated configuration
    eval_graph = build_eval_graph(
        csv_dir=validated_config.data.csv_dir,
        csv_filename=args.csv_filename,
        dataset_name=args.dataset_name,
        dataset_description=args.dataset_description,
        rag_retriever=structured_retriever,
        encoding=validated_config.report.encoding,
        top_k_matches=validated_config.retriever.top_k_matches,
        eval_llm=ragas_llm,
        summary_llm=summary_llm,
        report_generator=eval_report_generator,
    )

    # TODO: Consider implementing astream_events method in the evaluation graph for real-time streaming of evaluation results and interim reports.

    final_state = await eval_graph.ainvoke({"messages": []}) # type: ignore
    logger.info(f"Final evaluation state: {final_state}")

    # TODO: Will need to change print once move to front end implementation.

    print(final_state["llm_summary"])


if __name__ == "__main__":
    asyncio.run(main())
