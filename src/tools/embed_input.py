import logging
from typing import List, Literal, Union
from src.core.execution_service import ExecutionService
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

def embed_input(input:Union[Document, str], input_type: Literal["document", "query"]) -> List[float]: #type:ignore
    pass
    
