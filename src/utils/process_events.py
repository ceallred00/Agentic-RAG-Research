from typing import Literal, Iterator, List
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
import logging

logger = logging.getLogger(__name__)

#TODO: Current front end is CLI based. Future frontends (e.g., web) will need different rendering logic.

def process_events(events: Iterator[dict], thinking_flag: Literal[True, False] = False):
    """
    Parses and renders the stream of events from the agent application to the console.

    This function acts as the "Frontend" or "Presentation Layer" for the CLI. It iterates 
    over the generator, extracts the most recent message from each state update, and 
    prints it in a user-friendly format.

    Args:
        events (Iterator[dict]): A generator yielding state updates from `application_streamer`.
        thinking_flag (bool): If True, parses and prints the internal "thinking" blocks 
                              from the model (if available). Defaults to False.

    Event Structures Handled:
    -------------------------
    1. Standard Node Update:
       {
           'node_name': {
               'messages': [AIMessage(content="...", additional_kwargs={}, ...), ...] 
           }
       }
       
    2. Error Event (yielded by application_streamer on crash):
       {
           'error': "Error description string"
       }

    Processing Logic:
    -----------------
    - Error Handling: Checks for the 'error' key immediately and stops processing if found.
    - Message Extraction: Only processes the *last* message in the 'messages' list for each update.
    - Rendering: Differentiates output based on message type (AIMessage vs ToolMessage) 
      and content (Thinking blocks vs. Final response).
    """
    
    for event in events:
        if "error" in event:
            print(f"Error in application_streamer: {event['error']}")
            logger.error(f"Error in application_streamer: {event['error']}")
            return 
        
        for node_name, values in event.items():
            messages: List[BaseMessage] = values.get("messages", [])
            # Get the last message from the updated state
            if not messages:
                continue

            last_message = values["messages"][-1]
            content = last_message.content

            if isinstance(last_message, AIMessage):
                if thinking_flag and isinstance(content, list):
                        for content_block in content:
                            if (isinstance(content_block, dict) and "thinking" in content_block):
                                print(f"\nAI Thoughts: {content_block['thinking']}")
            
                if isinstance(content, str) and content.strip(): # Ensures the content isn't just a plan string
                    print(f"\nAI: {last_message.text}")
                elif isinstance(content, list):
                    if last_message.text:
                        print(f"\nAI: {last_message.text}")
            
                if last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        print(f" Calling: {tc['name']} with {tc['args']}")
                        logger.info(f" Calling: {tc['name']} with {tc['args']}")

            elif isinstance(last_message, ToolMessage):
                print(f"\n TOOL RESULT: {last_message.content}")
                logger.info(f"\n TOOL RESULT: {last_message.content}")