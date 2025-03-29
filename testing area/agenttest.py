from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from testingvdb import db

# Getting the access keys for the llms
from dotenv import load_dotenv
import os

load_dotenv()

groq_llm = ChatGroq(
    model="llama3-8b-8192",
    api_key = os.getenv('GROQ_API_KEY')
)
# Basically, it's the "aplication's logs"
search = db.similarity_search('Quem é Ricken?', k=1)
res = search[0].page_content
messages = [
    SystemMessage('Você é um Agente de Inteligência Artificial que deverá responder perguntas, de forma breve (até 2 linhas), de acordo com um contexto'),
    SystemMessage('Contexto: '+ res
    )
]

while True:
    if (user_input := input('Digite uma mensagem: ')) == 'quit':
        break
    else:
        messages.append(HumanMessage(content=user_input))
        llm_res = groq_llm.invoke(messages)
        messages.append(AIMessage(llm_res.content))
        print(llm_res.content)
