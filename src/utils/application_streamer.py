import logging
from typing import Literal, Iterator
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def application_streamer(
    application: CompiledStateGraph,
    user_input: str,
    configuration: RunnableConfig,
    stream_mode: Literal["values", "updates"] = "updates",
) -> Iterator[dict]:
    """
    Wraps the application stream call.

    The .stream() method runs the graph one node at a time, streaming the intermediate state.

    Args:
        application (Compiled StateGraph): The compiled LangGraph runnable.
        user_input (str): The text input from the user.
        configuration (RunnableConfig): Configuration dict (e.g., thread_id)
        stream_mode (str): "updates" (change in state) or "values" (full state). Defaults to "updates"

    Yields:
        dict: A dictionary containing either:
            - Node Updates: `{'node_name': {'messages': [...]}}`
            - Error info: `{'error': 'Error description'}`

    Exception Handling:
        If an exception occurs during execution (e.g., Network Error, GraphRecursionError), it does NOT
        raise the exception.

        Instead, it:
        1. Logs the full traceback.
        2. Yields a dictionary `{"error": str(e)}`

        This prevents the UI or CLI from crashing, allowing the frontend to display a user-friendly error message.
        It also keeps the session alive, allowing the user to retry immediately.
            The user will see the error message, then see User:
    """
    try:
        yield from application.stream(
            {"messages": [HumanMessage(content=user_input)]},  # Appends the user_input to the state.
            config=configuration,
            stream_mode=stream_mode,
        )
    except Exception as e:
        logger.error(f"Error in application_streamer: {e}", exc_info=True)
        yield {"error": str(e)}
