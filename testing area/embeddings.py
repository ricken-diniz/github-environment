from sentence_transformers import SentenceTransformer

# Carregar o modelo
modelo = SentenceTransformer('BAAI/bge-small-en')

# Frases de exemplo
frases = ["Exemplo de frase 1.", "Exemplo de frase 2."]

# Gerar embeddings
embeddings = modelo.encode(frases)

print(embeddings)