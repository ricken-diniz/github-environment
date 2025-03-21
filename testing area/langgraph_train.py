import os
from dotenv import load_dotenv
load_dotenv('.env')

# os.environ("LANGCHAIN_PROJECT") = "Demo LangGraph 001"
'''
    This is responsible to link to a langsmith in the fucture
'''
from typing import Annotated, Union

from typing_extensions import TypedDict
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    agent_outcome: Union[AgentAction, AgentFinish, None]
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

groq_llm = ChatGroq(
    model="llama3-8b-8192",
    api_key = os.getenv('GROQ_API_KEY')
)
llm_with_tools = groq_llm.bind_tools(tools)

def run_agent(data):
    agent_outcome = llm_with_tools.invoke(data)
    return {"agent_outcome": agent_outcome}

def execute_tools(data):
    agent_action = data['agent_outcome']
    output = llm_with_tools.invoke(agent_action)
    print(f"The agent action is {agent_action}")
    print(f"The tool result is: {output}")
    return {"intermediate_steps": [(agent_action, str(output))]}

def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
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
    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"])

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)