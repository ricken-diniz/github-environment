
from pymilvus import model

# Carregar o modelo
sparse_embedding_model = model.sparse.SpladeEmbeddingFunction(
        'naver/splade-cocondenser-selfdistil', 
        device="cpu"
    )

print(sparse_embedding_model.encode_documents(['Olá']))