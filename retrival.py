from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Puedes usar mistral, phi3 o llama2 también


db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


retriever = db.as_retriever()


# 4. Crear el modelo LLM
llm = ChatOllama(model="mistral")  # modelo local


# 5. Crear la cadena de RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# 6. Probar
while True:
    query = input("Pregunta: ")
    if query.lower() in ["salir", "exit", "quit"]:
        break

    result = qa_chain({"query": query})
    print("\n💬 Respuesta:", result["result"])

    print("\n📄 Fuente(s):")
    for doc in result["source_documents"]:
        print("-", doc.metadata["source"])