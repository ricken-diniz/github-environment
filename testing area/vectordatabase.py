'''
    ÁREA EM CONSTRUÇÃO

    - Lógica sendo pensada meticulosamente para diminuir os custos
    com recursos e processamento, visto que a cada troca de requisições
    o código será processado.
    
'''
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType, model
from langchain_milvus import BM25BuiltInFunction

client = MilvusClient("milvus_demo.db") # Client for a local database, this can be made with cluster

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 128},
)

index_params.add_index(
    field_name="sparse",
    index_name="sparse_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={"inverted_index_algo": "DAAT_MAXSCORE"},
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
client.drop_collection(collection_name="hybrid_search_collection")
if not client.has_collection(collection_name="hybrid_search_collection"):
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR)

    client.create_collection(
        collection_name="hybrid_search_collection",
        schema=schema,
        index_params=index_params
    )


# Insert more docs in another subject.
docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]
vectors = embedding_docs(docs)
data = [
    {"id": 3 + i, "text": docs[i],"sparse": [vectors['sparse'][i]], "dense":  vectors['dense'][i]} # !!!IMPORTANT!!! Is important encapsulate sparse vectors
    for i in range(len(vectors))
]

client.insert(collection_name="hybrid_search_collection", data=data)

# This will exclude any text in "history" subject despite close to the query vector.
res = client.search(
    collection_name="hybrid_search_collection",
    data=dense_embedding_model.encode_queries(["tell me AI related information"]),
    limit=2,
    output_fields=["text"],
)

print(res)