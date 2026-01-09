""" Defines the generic AgentState class used to track the state of agents in the system. """ 

from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # Preserve the state by appending the new message to the existing list of messages.