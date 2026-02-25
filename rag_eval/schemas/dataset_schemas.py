from pydantic import BaseModel, Field
from typing import Annotated

class QAPair(BaseModel):
    """
    Represents a question-answer pair extracted from a document, where the question is 
    generated based on the content of the document and the answer is the relevant 
    information that should be retrieved by the RAG system.
    """
    question: Annotated[str, Field(min_length=1, description="A natural language question generated from the document content.")]
    ground_truth: Annotated[str, Field(min_length=1, description="The expected correct answer that should be retrieved by the RAG system for the given question.")]

class QAPairList(BaseModel):
    """
    Represents a list of question-answer pairs.
    This is used to structure the output of the dataset generation process,
    where multiple question-answer pairs can be generated from a single document.
    """
    qa_pairs: Annotated[list[QAPair], Field(description="A list of question-answer pairs generated from the document content.")]

# DatasetRow represents a single row in the generated evaluation CSV.
# It extends QAPair by adding the source document path, which is known at
# generation time but is not part of the LLM output.
class DatasetRow(BaseModel):
    """
    Represents a single row in the evaluation dataset CSV, combining the
    LLM-generated question and ground truth answer with the source document path.
    """
    question: str
    ground_truth: str
    source: str