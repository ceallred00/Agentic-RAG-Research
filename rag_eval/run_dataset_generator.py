"""
CLI entry point for generating RAG evaluation datasets from the
UWF Public Knowledge Base.

Usage:
    python -m rag_eval.generate_dataset \
        --sample-size 20 \
        --output-filename dataset_v1.csv

Run with --help for the full argument list.
"""
import argparse
import logging
import sys

from core.execution_service import ExecutionService
from core.logging_setup import setup_logging
from constants import (
    UWF_PUBLIC_KB_PROCESSED_DATE_DIR,
    RAG_EVAL_DATA_DIR,
)
from rag_eval.dataset_generator import DatasetGenerator

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAG evaluation dataset from the UWF Public Knowledge Base.",
        # Automatically appends (default: X) to each help string.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        # Eden AI model format: provider/model-name.
        # See available models: https://docs.edenai.co/v3/how-to/llm/chat-completions#available-models
        default="anthropic/claude-haiku-4-5",
        help="Eden AI model name in provider/model-name format.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="Number of documents to randomly sample from the knowledge base.",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=1,
        help="Number of question-answer pairs to generate per document.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        required=True,
        help="Name of the output CSV file (e.g. dataset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAG_EVAL_DATA_DIR),
        help="Directory to save the generated CSV.",
    )
    parser.add_argument(
        "--kb-dir",
        type=str,
        default=str(UWF_PUBLIC_KB_PROCESSED_DATE_DIR),
        help="Knowledge base directory containing markdown files.",
    )
    parser.add_argument(
        "--min-doc-length",
        type=int,
        default=200,
        help="Minimum document character length to include in sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature. 0.0 = deterministic output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(
        f"Starting dataset generation | model={args.model} | temperature={args.temperature} | "
        f"sample_size={args.sample_size} | n_questions={args.n_questions} | "
        f"output={args.output_filename}"
    )

    execution_service = ExecutionService()
    llm = execution_service.get_eden_ai_client(model_name=args.model, temperature=args.temperature)

    generator = DatasetGenerator(
        llm=llm,
        output_dir=args.output_dir,
        kb_dir=args.kb_dir,
        min_doc_length=args.min_doc_length,
    )

    try:
        output_path = generator.generate_dataset(
            sample_size=args.sample_size,
            output_filename=args.output_filename,
            n_questions=args.n_questions,
        )
        logger.info(f"Dataset successfully saved to: {output_path}")
        print(f"\nDataset saved to: {output_path}")
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
