from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

default_ef = embedding_functions.DefaultEmbeddingFunction()

client = PersistentClient('./meu_banco')

# collection = client.create_collection(name="my_collection", embedding_function=default_ef)
collection = client.get_collection(name="my_collection", embedding_function=default_ef)

def ler_txt_e_retorna_texto_em_document():
    print(f">>> REALIZANDO A LEITURA DO TXT EXEMPLO")
    # lendo o txt com o texto exemplo e criando o Document:
    lista_documentos = TextLoader('exemplo_texto.txt', encoding='utf-8').load()

    print("Texto lido e convertido em Document")
    print(lista_documentos)
    print("-----------------------------------")
    return lista_documentos
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

# collection.add(
#     documents=["lorem ipsum...", "doc2", "doc3"],
#     metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
#     ids=["id1", "id2", "id3"]
# )

res = collection.query(
    query_texts=['lorem ipsum...'],
    n_results=1,
    where={"metadata_field": "is_equal_to_this"}
)

print(res)