from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA


###########################################
########################################### 1. Cargar y dividir el texto


# defining the document path
text_path = f"/home/martin/Documents/git_proyects/RAG-sentiment-analysis/docs/causalAI.pdf"

# loading the document
loader = PyPDFLoader(text_path)
documents = loader.load()

######### splitingthe document in chunks

# defining the text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # this takes 500 characters and averlap 50 characters between each element of the list

docs_split = splitter.split_documents(documents)


###########################################
########################################### 2. Crear embeddings con modelo local

""" 
para cargarcar mistral y usarlo como modelo de embedings, primero se necesita descargar (lo mismo seria con
cualquier otro modelo en ollama
"""

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Puedes usar mistral, phi3 o llama2 también


# 3. Crear vectorstore (ChromaDB)
db = Chroma.from_documents(docs_split, embedding=embeddings, persist_directory="./chroma_db")
