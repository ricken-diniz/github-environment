import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

# Definindo o modelo de embedding
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPEN_API_KEY'))

# Função: Divide o documento em partes menores (chunks) de tamanho 1000 caracteres, com prioriade para a quebra por
# paragrafo.
def divide_texto(lista_documento_entrada):
    print(f">>> REALIZANDO A DIVISAO DO TEXTO ORIGINAL EM CHUNKS")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(lista_documento_entrada)  # usado split_documents dado que a entrada é uma lista de documentos:
    i = 0
    for pedaco in documents:
        print("--" * 30)
        print(f"Chunk: {i}")
        print("--" * 30)
        print(pedaco)
        print("--" * 30)
        i += 1
    return documents

# Cria o banco de dados vetorial, gerando os embeddings dos documentos
def cria_banco_vetorial_e_indexa_documentos(documentos):
    print(f">>> REALIZANDO INDEXAÇÃO DOS CHUNKS NO BANCO VETORIAL")
    # Cria o banco de dados vetorial, gerando os embeddings dos documentos
    # Adicionar os chunks no banco em lote
    Chroma.from_documents(documentos, collection_name="nome_colecao", embedding=embeddings_model, persist_directory="./meu_banco")



def ler_txt_e_retorna_texto_em_document():
    print(f">>> REALIZANDO A LEITURA DO TXT EXEMPLO")
    # lendo o txt com o texto exemplo e criando o Document:
    lista_documentos = TextLoader('exemplo_texto.txt', encoding='utf-8').load()

    print("Texto lido e convertido em Document")
    print(lista_documentos)
    print("-----------------------------------")
    return lista_documentos


def conecta_banco_vetorial_pre_criado():
    vector_store_from_client = Chroma(
        persist_directory="./meu_banco",
        collection_name="nome_colecao",
        embedding_function=embeddings_model,
    )
    return vector_store_from_client


# Verifica se o diretório "./meu_banco" não existe
if not os.path.exists("./meu_banco"):
    print("O diretório './meu_banco' não existe... realizando a indexação")
    texto_completo_lido = ler_txt_e_retorna_texto_em_document()
    divide_texto = divide_texto(texto_completo_lido)
    cria_banco_vetorial_e_indexa_documentos(divide_texto)
else:
    print("O diretório './meu_banco' já existe. Pulando a criação do banco vetorial.")

# Conectando ao banco vetorial pre criado com os dados indexados:
db = conecta_banco_vetorial_pre_criado()

# Agora podemos trabalhar com o banco uma vez que ele está com os dados já indexados.

query = "Na expansão da inteligência artificial quais questões importantes são levantadas?"
pedacoes_retornados = db.similarity_search(query, k=2)


# Total de docs retornados
print("Total de pedaços. Deve ter o valor de 'K':")
print(len(pedacoes_retornados))
# Exibir o conteúdo do primeiro documento retornado
# Imprimindo os pedaços retornados do banco:
i=0
for elm in pedacoes_retornados:
    print(f"------ chunk {i} -------")
    print(pedacoes_retornados[i].page_content)
    print("--------------------")
    i+=1