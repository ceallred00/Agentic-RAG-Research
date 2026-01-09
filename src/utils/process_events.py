from typing import Literal, Iterator, List
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

#TODO: Add logging and error handling.

def process_events(events: Iterator[dict], thinking_flag: Literal[True, False] = False):
    """
    Events is a generator object, holding a dictionary.

    Each event is a dictionary representing the updated state after each node execution.

    Event dictionary:
        Key: Node Name (Unique to whichever node is being executed)
        Value: Update made to the Agent State [dict]
    
    AgentState dictionary:
        Key: messages
        Value: List of message objects (HumanMessage, AIMeessage, ToolMessage, etc.)

    Example event structure:

    {'base_agent': 
        {'messages': 
            [AIMessage(content="...", additional_kwargs={}, ...)] 
        }
    }
    """
    for event in events:
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

            elif isinstance(last_message, ToolMessage):
                print(f"\n TOOL RESULT: {last_message.content}")