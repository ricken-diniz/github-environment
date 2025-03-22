import os
from dotenv import load_dotenv
load_dotenv('.env')

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.tools import tool
import random

@tool
def random_number_maker(input:str)->str:
    '''Returns a number between 1 and 10000'''
    return random.randint(0, 10000)

tool = TavilySearchResults(max_results=2, api_key=os.getenv('TAVILY_API_KEY'))
tools = [tool,random_number_maker]

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)  

from langchain_groq import ChatGroq

groq_llm = ChatGroq(
    model="llama3-8b-8192",
    api_key = os.getenv('GROQ_API_KEY')
)
llm = groq_llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)


import json

from langgraph.prebuilt import ToolNode, tools_condition


tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)


#Compiler
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

memory = MemorySaver()
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
workflow = graph_builder.compile(checkpointer=memory)
def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    for event in workflow.stream({"messages": [{"role": "user", "content": user_input}]}, config=config, stream_mode='values'):
        print(event)
        if any(isinstance(item, AIMessage) for item in event["messages"]):
            print("Assistant:", event["messages"][-1].content)


from IPython.display import Image

try:
    with open("graph.png", "wb") as f:
        f.write(Image(workflow.get_graph().draw_mermaid_png()).data)

except Exception:
    # This requires some extra dependencies and is optional
    pass

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

