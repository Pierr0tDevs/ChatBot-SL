import csv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Configurar tu clave de API
os.environ["GOOGLE_API_KEY"] = "AIzaSyDgOIp7zdj14G0xXvtMti72nE7_Bhrhtss"

# Inicializar embeddings y base Chroma
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Leer productos desde CSV limpio
with open("productos_limpios.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";")
    documentos = []
    metadatas = []

    for row in reader:
        texto = f"""{row['nombre']}

{row['descripcion']}"""
        url = row['link']
        documentos.append(texto)
        metadatas.append({"source": url})

# Agregar productos a la base Chroma
vector_store.add_texts(texts=documentos, metadatas=metadatas)

print("✅ Base de datos cargada correctamente.")
