from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

import os

import threading


def chonker():
    documents = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            pdf_path = "./docs/" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith(".docx") or file.endswith(".doc"):
            doc_path = "./docs/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            text_path = "./docs/" + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    print("chunked documents")
    return chunked_documents


# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
def createvectorDB(docs, db):
    print(docs)
    print("initializing vector")
    vectordb = Chroma.from_documents(
        docs,
        embedding=OllamaEmbeddings(model="mistral", temperature=0.9),
        persist_directory="./data",
    )
    db = vectordb
    print("vector initialized")
    vectordb.persist()
    print("created vectorDB")
    return


def createQAchain(vectordb):
    """
    Creates a conversational retrieval chain for question-answering.

    Returns:
        qa_chain (ConversationalRetrievalChain): The created QA chain.
    """
    print("creating QA chain")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model="mistral", temperature=0.9, verbose=True),
        retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
    )
    print("created QA chain")
    return qa_chain


chat_hist = []
db = None
vectorThread = threading.Thread(target=createvectorDB, args=(chonker(), db))
vectorThread.start()

vectorThread.join()
if db:
    chain = createQAchain(db)
else:
    print("Error creating vectorDB")
while True:
    if db != None:
        userInp = input("[+] Write your query here: \n")
        if userInp == "q":
            print("exiting...")
            break
        anws = chain.invoke({"question": userInp, "chat_history": chat_hist})
        print("[Mistral]-> " + anws["answer"])
        chat_hist.append((userInp, anws))
        print("\n\n")
        print("chat history: ")
        print(chat_hist)
