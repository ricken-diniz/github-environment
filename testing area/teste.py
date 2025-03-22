"""
    NOTAS:
        .A classe State serve para manipular as respostas das LLMs, o compilador de grafo de estados é
        responsável por cuidar dessa parte de invocar as LLMs e as ferramentas, gerando assim um fluxo de
        respostas.

        .O StateGraph é responsável por criar um grafo de estados que será utilizado para criar um fluxo de
        respostas e condições, tratando essas até chegar ao objetivo final desejado pelo prompt do usuário.

        .A classe TypedDict é uma classe que serve para definir um dicionários, usamos ela como base para a
        classe State, usando conceitos de herança e polimorfismo. No mesmo intuito, o decorator Annotated é
        usado para facilitar a manipulação dos dados desses dicionários. Outros decorators: add_messages.

        .O decorator tool é usado para definir funções que serão utilizadas como ferramentas para manipular
        as respostas das LLMs, refinando-as até chegar ao objetivo final.

        .A classe ToolMessage é uma classe que serve para manipular as mensagens das ferramentas, sendo
        responsável por gerar as mensagens de saída.
"""





import os
from dotenv import load_dotenv
load_dotenv('.env')

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

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


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

workflow = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in workflow.stream({"messages": [{"role": "user", "content": user_input}]}):
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