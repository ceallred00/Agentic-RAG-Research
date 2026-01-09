from typing import Literal, Iterator
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

#TODO: Add logging and error handling here.

def application_streamer(
        application: CompiledStateGraph, 
        user_input: str, 
        configuration: RunnableConfig, 
        stream_mode: Literal["values", "updates"] = "updates") -> Iterator[dict]:
    """
    Wraps the application stream call. 

    The .stream() method runs the graph one node at a time, streaming the intermediate state.

    Args:
        application: 
        user_input:
        configuration:
        stream_mode:
         
    """

    return application.stream(
        {"messages": [HumanMessage(content=user_input)]}, # Appends the user_input to the state.
        config = configuration, 
        stream_mode = stream_mode
    )

    