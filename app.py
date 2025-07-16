from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import google.generativeai as genai
import markdown
import re

# ✅ Crear app Flask y configuración de sesión
app = Flask(__name__)
app.secret_key = ""  # Cambiá esto por algo seguro
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ✅ Configurar API Key de Google
load_dotenv()

# Usar variables desde .env
app.secret_key = os.getenv("SECRET_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ✅ Embeddings y vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ✅ Formatear documentos recuperados
def format_docs(docs):
    return "\n\n".join(
        [f"{doc.page_content}\nFuente: {doc.metadata.get('source', 'Desconocido')}" for doc in docs]
    )

# ✅ Normalizar precios tipo $87.99 → 87.99 USD
def normalizar_precios(texto):
    return re.sub(r"\$([0-9]+(?:\.[0-9]{1,2})?)\b(?!\s*USD)", r"\1 USD", texto)

# ✅ Prompt con Markdown y contexto de productos
prompt_template = """
Eres un asistente experto y amigable de **ShiGa Labs**, una tienda especializada en componentes para PC.

Tu tarea es ayudar al cliente usando **solo** la información disponible en los siguientes productos:

---

{context}

---

### Pregunta del cliente:
{question}

### Instrucciones:
- Responde en español, de forma clara y amable.
- Presentate como el bot de ShiGa Labs.
- Usá formato **Markdown** para la respuesta.
- Si recomendás un producto, explicá por qué es buena opción.
- Incluir el **nombre**, **precio** y un **[enlace al producto](URL)** si está disponible.
- Si no hay una opción adecuada, escribí:  
  _"No he encontrado un producto que se ajuste a tu consulta..."_
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ✅ RAG Chain
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ✅ Página principal con historial por sesión
@app.route("/", methods=["GET", "POST"])
def index():
    if "historial" not in session:
        session["historial"] = []

    if request.method == "POST":
        pregunta = request.form["pregunta"]
        try:
            respuesta_bruta = rag_chain.invoke(pregunta)
            respuesta_limpia = normalizar_precios(respuesta_bruta)
            respuesta_md = markdown.markdown(respuesta_limpia)

            session["historial"].append({
                "pregunta": pregunta,
                "respuesta": respuesta_md
            })
            session.modified = True
        except Exception as e:
            session["historial"].append({
                "pregunta": pregunta,
                "respuesta": f"<span class='text-danger'>❌ Error: {e}</span>"
            })
            session.modified = True

    return render_template("index.html", historial=session["historial"])

# ✅ Ruta para borrar el historial
@app.route("/reset", methods=["POST"])
def reset():
    session.pop("historial", None)
    return redirect(url_for("index"))

# ✅ Iniciar servidor
if __name__ == "__main__":
    app.run(debug=True)
