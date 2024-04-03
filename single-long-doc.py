from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain.chains import  RetrievalQA

#carregar o documento
loader = PyPDFLoader("Resume.pdf")
docs = loader.load()    

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(docs)

vectordb = Chroma(
    docs,
    # embedding=OllamaEmbeddings(),
    persist_directory="./data"
)
vectordb.persist()

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="mistral", temperature=0.9),
    retriever=vectordb,
    return_source_documents=True
)

res = qa_chain.invoke({"input_documents": docs, "question": "What is your name?"})
print(res)