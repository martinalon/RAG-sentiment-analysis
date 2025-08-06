from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



# Definir el prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant that responds accurately based on the context provided.

Contexto:
{context}

Pregunta:
{question}

If you don't know the answer, answer with "I don't know", don't make anything up.
""",
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Puedes usar mistral, phi3 o llama2 también


db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


retriever = db.as_retriever(search_kwargs={"k": 5})


# 4. Crear el modelo LLM
llm = ChatOllama(model="mistral")  # modelo local


# 5. Crear la cadena de RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents= True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
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