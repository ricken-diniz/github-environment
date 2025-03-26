"""
    With this extension for langchain, we can get access a milvus databases
    using langchain objects! But, its better to structure the milvus data base
    using the Milvus model itself.
    Even so, the langchain_miluvs is so great for create a collections and to adding 
    datas in the database!
"""


import os
from dotenv import load_dotenv
load_dotenv('.env')

from langchain_milvus import Milvus, BM25BuiltInFunction
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction


URI = os.getenv('VECTOR_DB_URI')

docs = [
    
]

dense_embedding_model1 = SentenceTransformer('all-MiniLM-L6-v2')
dense_embedding_model2 = SentenceTransformerEmbeddingFunction(
    model_name = "hkunlp/instructor-xl",
    device='cpu'
)
sparse_embedding_model = BM25BuiltInFunction()

# Inserting datas in milvus db
vector_store = Milvus.from_documents(
    documents=[Document(page_content='Hello World!')],
    embedding=dense_embedding_model2,
    # `dense` is for dense embeddings, `sparse` is the output field of BM25 function
    vector_field=["dense"],
    collection_name="hybrid_search_collection",
    connection_args={
        "uri": URI,
    },
    consistency_level="Strong",
    drop_old=True,
)



