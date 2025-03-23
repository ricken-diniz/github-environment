from sentence_transformers import SentenceTransformer, util

# Carregar o modelo
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# Lista de frases
frases = [
    "O cachorro está brincando no parque.",
    "A lua brilha à noite.",
    "O gato dorme no sofá.",
    "Hoje o tempo está ensolarado.",
    "Pizza de calabresa custa 10 dólares.",
    "Pizza de carne, de frango e de quatro queijos."
]

# Gerar embeddings para todas as frases
embeddings = modelo.encode(frases, convert_to_tensor=True)

# Frase de busca
consulta = "Quanto custa o carro?"
emb_consulta = modelo.encode(consulta, convert_to_tensor=True)

# Calcular similaridades com todas as frases
similaridades = util.pytorch_cos_sim(emb_consulta, embeddings)

# Obter índice da frase mais similar
indice_mais_similar = similaridades.argmax()
print(f"Frase mais similar: {frases[indice_mais_similar]}")