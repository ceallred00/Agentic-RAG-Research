import csv
import json
import logging
import random
import datetime
from typing import Union, List, Dict
from pathlib import Path
import frontmatter
from langchain_openai import ChatOpenAI # Type hint for LLM parameter
import openai 

from constants import UWF_PUBLIC_KB_PROCESSED_DATE_DIR, RAG_EVAL_DATA_DIR
from rag_eval.schemas.dataset_schemas import QAPairList, DatasetRow

logger = logging.getLogger(__name__)

class DatasetGenerator:
    def __init__(self, llm: ChatOpenAI, output_dir: Union[str, Path] = RAG_EVAL_DATA_DIR, kb_dir: Union[str, Path] = UWF_PUBLIC_KB_PROCESSED_DATE_DIR,min_doc_length: int = 200):
        self.kb_dir = Path(kb_dir)
        self.llm = llm
        self.structured_llm = llm.with_structured_output(QAPairList)
        self.output_dir = Path(output_dir)
        self.min_doc_length = min_doc_length
    
    def list_documents(self) -> List[Path]:
        """
        Lists all markdown documents in the knowledge base directory.
        
        Raises a FileNotFoundError if the directory does not exist.
        """
        if not self.kb_dir.exists():
            error_msg = f"Knowledge base directory {self.kb_dir} does not exist."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        docs = list(self.kb_dir.glob("*.md"))
        logger.info(f"Found {len(docs)} documents in knowledge base directory {self.kb_dir}")
        return docs
    def filter_documents(self, documents: List[Path]) -> List[Dict[str, str]]:
        """
        Loads each document, strips YAML frontmatter,
        and filters out documents that are shorter than the specified minimum length.

        Extracts the content of each document and the full source path.

        Skips documents that cannot be processed and logs any errors.

        Returns a list of dictionaries containing the content and source path of each 
        valid document, extracted from the YAML frontmatter if available. Fallsback
        to the file path if no "path" field is found in the frontmatter.

        Output format:
        [
            {
                "content": "The content of the document without frontmatter.",
                "source": "UWF Public Knowledge Base / Wireless Networks"
            }, 
            ...
        ]
        """ 
        filtered_docs = []
        logger.info(f"""Filtering {len(documents)} documents based on minimum length 
                    of {self.min_doc_length} characters""")
        for doc in documents:
            try:
                # Load doc from Path
                loaded_doc = frontmatter.load(doc) #type: ignore
                # Strip frontmatter and get content
                content = loaded_doc.content.strip()
                # Filter based on length
                if len(content) < self.min_doc_length:
                    logger.info(f"""Filtering out document {doc} due to insufficient 
                                length ({len(content)} chars)""")
                    continue
                returned_doc = {
                    "content": content,
                    "source": loaded_doc.get("path", str(doc))
                }                   
                filtered_docs.append(returned_doc)        
            except Exception as e:
                logger.error(f"Error processing document {doc}: {e}")
                continue
        logger.info(f"Returning {len(filtered_docs)} documents after filtering")
        return filtered_docs
    def sample_documents(self, documents: List[Dict[str, str]], 
                         sample_size: int) -> List[Dict[str, str]]:
        """
        Randomly samples a specified number of documents from the list.

        If the requested sample size exceeds the number of available documents, 
        returns all documents and logs a warning.

        No seed is set for randomness to ensure variability across runs.
        """
        if sample_size > len(documents):
            logger.warning(f"""Requested sample size {sample_size} exceeds available 
                           documents {len(documents)}. Returning all documents.""")
            return documents
        # Samples without replacement to ensure unique documents in the sample.
        sampled_docs = random.sample(documents, sample_size)
        logger.info(f"""Randomly sampled {len(sampled_docs)} documents for 
                    dataset generation""")
        return sampled_docs
    def generate_qa_pairs(self,
                          content: str,
                          source: str,
                          n_questions: int = 1) -> List[DatasetRow]:
        """
        Uses the LLM to generate n question-answer pairs based on the content of
        the document.

        The number of questions generated per document can be specified.
        Defaults to 1 question per document to balance dataset size and generation time.

        Returns a list of DatasetRow objects containing the question, ground truth
        answer, and source document path.
        """
        prompt = f"""
        Given the following document content, generate {n_questions} question-answer
        pairs that could be used to evaluate a Retrieval-Augmented Generation (RAG)
        retriever for a UWF academic advisor chatbot.

        The questions must be phrased as practical, realistic queries that a UWF
        student, faculty member, or staff member might actually ask an AI academic
        advisor — for example, questions about university policies, procedures,
        resources, or student life. Avoid definitional or textbook-style questions
        (e.g. "What is the definition of X?"). Prefer questions like "How do I...",
        "What are my options for...", or "Who should I contact about...".

        The answers must be concise, factual responses drawn exclusively from the
        content provided below. Do not use any outside knowledge. If the document
        does not contain enough information to answer a question, do not generate
        that pair.

        Document:
        {content}
        """
        try:
            # Returns a QAPair Pydantic model with a list of generated QA pairs.
            response = self.structured_llm.invoke(prompt)

            qa_pairs = []

            # Slice to n_questions to enforce the requested count. The LLM may return
            # more pairs than requested when the schema has no upper bound — slicing
            # ensures code-level enforcement rather than relying on the prompt or schema.
            for qa_pair in response.qa_pairs[:n_questions]: #type: ignore
                qa_pairs.append(DatasetRow(
                    question=qa_pair.question,
                    ground_truth=qa_pair.ground_truth,
                    source=str(source)
                ))
            
            # Ensure that at least one QA pair was generated.
            # If not, raise an error to avoid including empty samples in the dataset.
            if not qa_pairs:
                error_msg = f"""No question-answer pairs were generated for 
                document {source}."""
                logger.error(error_msg)
                raise ValueError(error_msg)
            return qa_pairs
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    def generate_dataset(self, sample_size: int, output_filename: str, n_questions: int = 1) -> Path:
        """
        Main method to generate the dataset. It orchestrates the entire process:
        - Lists documents in the knowledge base directory.
        - Filters documents based on minimum length.
        - Randomly samples sample_size documents.
        - Generates n_questions question-answer pairs per document using the LLM.
        - Saves the generated dataset to a CSV file in the output directory.

        Args:
            sample_size: Number of documents to randomly sample from the knowledge base.
            output_filename: Base name of the output CSV file. The current date is
                automatically prepended in YYYYMMDD_ format (e.g. passing
                "dataset_batch_1.csv" produces "20260225_dataset_batch_1.csv").
            n_questions: Number of question-answer pairs to generate per document. Defaults to 1.

        Returns:
            The path to the generated CSV file containing the dataset.
        """
        # List documents
        all_docs = self.list_documents()
        # Filter documents
        filtered_docs = self.filter_documents(all_docs)
        # Sample documents
        sampled_docs = self.sample_documents(filtered_docs, sample_size)
        # Generate QA pairs
        qa_pairs: List[DatasetRow] = []
        for doc in sampled_docs:
            try:
                # Returns a list of QA pairs for the document
                qa_pair = self.generate_qa_pairs(doc["content"], doc["source"], n_questions)
                # Extend main list with the generated pairs for the document
                qa_pairs.extend(qa_pair)
            except Exception as e:
                logger.error(f"Skipping document {doc['source']}: {e}")
                continue
        
        if not qa_pairs:
            error_msg = """No question-answer pairs were generated for any documents.
            Dataset generation failed."""
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        #  Check that output directory exists before attempting to save
        if not self.output_dir.exists():
            error_msg = f"Output directory {self.output_dir} does not exist."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Prepend the current date to the filename for automatic timestamping.
        dated_filename = f"{datetime.datetime.now().strftime('%Y%m%d')}_{output_filename}"
        output_path = self.output_dir / dated_filename

        # Ensure the output filename has a .csv extension.
        # If not, change the extension to .csv and log a warning.
        if output_path.suffix != ".csv":
            output_path = output_path.with_suffix(".csv")
            logger.warning(f"""Output filename {output_filename} does not have .csv 
                           extension. Saving to {output_path} instead.""")

        # Save to CSV
        try:
            logger.info(f"""Saving generated dataset with {len(qa_pairs)} question-answer 
                        pairs to {output_path}""")
            with open(output_path, mode='w', newline = "", encoding = 'utf-8') as f:
                writer = csv.DictWriter(f,
                                        fieldnames=list(DatasetRow.model_fields.keys()))
                writer.writeheader()
                writer.writerows([row.model_dump() for row in qa_pairs])
            
            return output_path

        
        except OSError as e:
            logger.error(f"Error writing to output file {str(output_path)}: {e}")
            raise

