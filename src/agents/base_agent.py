"""
This module is the working code for a base agent. 

Note that it is not the final version of the agent; rather, it is an in-progress version. 

Future iterations will include:
- Moving the tools into their own respective files and directories. 
- Moving the system prompt to the base_agent.yaml file.
- The graph configuration may get exported to the architecture YAML files, though that is to be decided at a later date.

"""

import logging
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from core.state import AgentState
from utils.application_streamer import application_streamer
from utils.process_events import process_events
from constants import FAKE_DEPARTMENT_ADVISORS
from google.genai.errors import ServerError

# from utils.architecture_diagram_generator import ArchitectureDiagramGenerator
# from constants import PROD_DIAGRAM_DIR

logger = logging.getLogger(__name__)

load_dotenv()

# TODO: Move tools to their own .py files and finalize.

@tool
def search_web(query: str):
    """
    Performs a libe web search. Use this ONLY if the perform_rag tool did not provide sufficient information for the user.
    """
    return f"Web search results for query: {query}"  # Example response

@tool
def vectorize_user_input(user_input: str) -> List[float]:
    """
    Converts raw user text into a numerical vector. This is a mandatory prerequisite for using the perform_rag tool.
    """
    return [-0.2, 0.1, 0.3, -0.1, 0.2]  # Example vectorized input

@tool
def perform_rag(user_query_vector: List[float]):
    """
    Searches the University of West Florida (UWF) knowledge base using a vector. 
    Use this to answer UWF-specific questions about policies, courses, or campus life.

    This tool should only be used after the user's input was vectorized using vectorize_user_input.
    """
    return "Relevant results from RAG process"  # Example response 

@tool
def draft_email(user_input: str, advisor_email: str, advisor_name: str, student_name: str, student_email: str):
    """ 
    Draft an email based on user input. This email will be used to communicate with the advisor. 
    The email should include the name of the student, the student's email address,the advisor's name, and a brief summary of the user's query. 
    The tone should be professional and courteous.
    
    This tool call should follow a call to search_for_advisor, which will help in identifying the advisor's email address and name.

    This tool call should be performed only if explicitly requested by the user, or if the user is not satisfied with the information provided by the RAG process or the web search.

    Once the email is drafted, return it to the user for review.
    """
    #TODO: Implement the email response structure
    pass

@tool
def send_email(email_content: str):
    """ 
    Send an email with the drafted content. 
    
    This tool call should follow a call to draft_email, only after the user has reviewed and approved the draft.
    The email should be sent to the advisor's email address, as identified using the search_for_advisor tool."""

    pass

@tool
def search_for_advisor(department: str):
    """
    Retrieves advisor contact details for a specific UWF department.
    
    If the department is not specified, you should prompt the user for their department.
    Once the advisor is found, return the email address of the advisor to the user.
    Then, ask the user if they would like you to draft and send an email to the advisor for further assistance.
    
    If the user agrees, draft an email using the draft_email tool. 
    """
    advisor_lookup = {key.lower(): value for key, value in FAKE_DEPARTMENT_ADVISORS.items()}

    advisor_info = advisor_lookup.get(department.lower())
    
    if not advisor_info: 
        return "I'm sorry, I couldn't find an advisor for that department. Please check the department name and try again."

    return f"Advisor found. Your advisor is {advisor_info['name']} at {advisor_info['email']}."  # Example response

@tool
def end_conversation() -> str:
    """ 
    Closes the active session. 
    
    Call this when the user says goodbye, indicates they are satisfied, or has no further questions.
    """
    return "Conversation ended"  # Example response


# Register the available tools
tools = [search_web, vectorize_user_input, perform_rag, draft_email, send_email, search_for_advisor, end_conversation]

# Initialize the model with the available tools
model = ChatGoogleGenerativeAI(model="gemini-pro-latest", include_thoughts=True).bind_tools(tools) # Bind available tools to the model

def base_agent(state: AgentState) -> AgentState: #type:ignore
    system_prompt = SystemMessage(content = f"""
        You are the UWF Student Assistant, a helpful AI dedicated to University of West Florida students.

        CORE WORKFLOW:
        1. INFORMATIONAL QUERIES:
        - Step A: Always call `vectorize_user_input` first.
        - Step B: Pass that vector into `perform_rag`.
        - Step C: Present the results. If the answer is incomplete, offer a `search_web` call.

        2. ADVISOR ASSISTANCE:
        - If the user needs to contact an advisor, call `search_for_advisor`.
        - If a department is missing, ask the user for it before calling the tool.
        - Once an advisor is found, offer to `draft_email`.

        3. EMAIL DISPATCH:
        - After drafting, show the content to the user. 
        - Call `send_email` ONLY after the user explicitly gives permission to send.

        4. CONVERSATION CLOSURE:
        - If the user indicates they are finished (e.g., "Thanks!", "Goodbye", "I'm all set"), 
            confirm they have no other questions, then call `end_conversation`.

        POLICIES:
        - Only provide information regarding the University of West Florida.
        - If you lack required tool arguments (like a department name), stop and ask the user for that specific information.
        - Be professional, empathetic, and concise.

        """)

    # Rules first + conversation history + most recent user message
    messages = [system_prompt] + list(state["messages"])
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Generate a response using the model
            response = model.invoke(messages)
            logger.info("Invoking model...")
            
            # Update the state by returning the old messages + the new user message + the new AI response
            return {"messages": [response]}
        except ServerError as e:
            if attempt == max_retries - 1:
                raise e


def should_continue(state: AgentState):
    """ 
    Conditional function to determine if the graph should continue or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and not last_message.tool_calls: # Look at last message to see if there are any tool calls. If not, end the graph.
        return "end" # Edge
    else: # If there are tool calls, continue the graph.
        return "continue" # Edge

    
def print_messages(messages):
    """ Print the message in a more readable format."""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("base_agent", base_agent)
graph.add_node("tool_node", ToolNode(tools))

graph.set_entry_point("base_agent")
graph.add_conditional_edges(
    source = "base_agent", 
    path = should_continue,
    path_map = {
        "continue": "tool_node",
        "end": END
    
    }
)
graph.add_edge("tool_node", "base_agent")

# Initialize Memory Checkpointer
memory = MemorySaver()

# Returns a CompiledStateGraph object
app = graph.compile(checkpointer=memory)


def run():
    """
    Run the base agent conversation loop.

    The graph is invoked when the user provides input.
    The graph will execute according to its architecture.
     
    The conversation continues until the user decides to exit, meaning that the graph may be invoked several times depending on the user's input.
    
    """
    print("\n The AI Assistant is ready to help you. Type 'exit' to end the conversation.")
    # A static thread_id simulates a persistent user session.
    config: RunnableConfig = {"configurable": {"thread_id": "123"}}

    # current_state is a StateSnapshot object
    current_state = app.get_state(config)

    # current_state.values is a dictionary.
    if not current_state.values or not current_state.values.get("messages"):
        # Kickstart AI
        initial_input = "Hi there!"

        events = application_streamer(application=app,
                                      user_input = initial_input, 
                                      configuration = config, 
                                      stream_mode = "updates")
        
        process_events(events = events, thinking_flag = False)

    while True:
        try: 
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the conversation.")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting the conversation.")
            break

        # Runs the graph one node at a time, streaming intermediate state.
        events = application_streamer(
            application = app, 
            user_input = user_input, 
            configuration = config, 
            stream_mode = "updates"
        )

        process_events(events = events, thinking_flag = True)

# base_agent_diagram = ArchitectureDiagramGenerator(PROD_DIAGRAM_DIR)
# base_agent_diagram.generate_graph_diagram("base_agent_v1.png", app)
    
if __name__ == "__main__":
    run()
