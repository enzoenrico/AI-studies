from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings import huggingface
from langchain.chains import RetrievalQA

# splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# vectordb
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Chroma

ollama = Ollama(model="mistral")

embeds = huggingface.HuggingFaceEmbeddings()

loader = WebBaseLoader("https://en.wikipedia.org/wiki/2023_Hawaii_wildfires")
data = loader.load()

vectorstore = Chroma.from_documents(documents=data, embedding=embeds)

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=10)
all_splits = splitter.split_documents(data)

chain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

while True:
    question = input("Ask a question: ")
    if question == "exit":
        break
    res = chain.invoke({"query": question})
    print('[MISTRAL]: ' + res['result'])
# question = "when was hawaiis request for a major disaster declaration approved?"
# print(chain.invoke({"query": question})['result'])
