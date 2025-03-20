import os
from dotenv import load_dotenv
load_dotenv('.env')

# os.environ("LANGCHAIN_PROJECT") = "Demo LangGraph 001"
'''
    This is responsible to link to a langsmith in the fucture
'''
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    intermediate_steps: Annotated[list, add_messages]

# Custom Tools

from langchain.tools import tool
import random

@tool('random_number', return_direct=True)
def random_number_maker(input:str)->str:
    '''Returns a number between 1 and 10000'''
    return random.randint(0, 10000)

@tool('numerical_order', return_direct=True)
def numerical_order(input:str)->str:
    '''Sorts numbers in ascending order'''
    input = list(input)
    input.sort()
    return "".join(input)

@tool('upper_case', return_direct=True)
def to_upper_case(input:str) -> str:
    '''Returns the string in uppercase'''
    return input.upper()

tools = [to_upper_case,numerical_order, random_number_maker]

#Graph Builder

workflow = StateGraph(State)

from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain_core.agents import AgentAction, AgentFinish

groq_llm = ChatGroq(
    model="llama3-8b-8192",
    api_key = os.getenv('GROQ_API_KEY')
)
llm_with_tools = groq_llm.bind_tools(tools)

def run_agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def execute_tools(data):
    print(data['messages'])
    agent_action = data['messages']
    output = llm_with_tools.invoke(agent_action)
    print(f"The agent action is {agent_action}")
    print(f"The tool result is: {output}")
    return {"intermediate_steps": [(agent_action, str(output))]}

def should_continue(data):
    if isinstance(data['messages'], AgentFinish):
        return "end"
    else:
        return "continue"
  
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge('action', 'agent')

graph = workflow.compile()

# run the graph

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break