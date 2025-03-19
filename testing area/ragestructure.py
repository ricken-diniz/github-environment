import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv('.env')

# os.environ("LANGCHAIN_PROJECT") = "Demo LangGraph 001"
'''
    This is responsible to link to a langsmith in the fucture
'''

import operator
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None] #set the agents' outcomes
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] #notes the agents actions

# Custom Tools

from langchain.tools import Tool, BaseTool, StructuredTool, tool
import random

@tool('random_number', return_direct=True)
def random_number_maker(input:str)->str:
    '''Returns a number between 1 and 100'''
    return random.randint(0, 100)

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

from langchain import hub
from langchain.agents import initialize_agent,AgentType
from langchain_groq import ChatGroq

prompt = hub.pull("hwchase17/openai-functions-agent")
groq_llm = ChatGroq(
    model="llama3-8b-8192",
    api_key = os.getenv('GROQ_API_KEY')
)

agent = initialize_agent(
    tools=tools,
    llm=groq_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": prompt}
)

# Creating Nodes

def run_agent(data):
    agent_outcome = agent.invoke(data)
    return {"agent_outcome": agent_outcome}

def execute_tools(data):
    agent_action = data['agent_outcome']
    output = agent.run(agent_action)
    print(f"The agent action is {agent_action}")
    print(f"The tool result is: {output}")
    return {"intermediate_steps": [(agent_action, str(output))]}

def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    else:
        return "continue"
    
# Graph

