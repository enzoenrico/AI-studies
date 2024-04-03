from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama


# load the document as before
loader = PyPDFLoader('Resume.pdf')
documents = loader.load()
print("loaded document")

# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
print("split document into chunks")

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
  documents,
  embedding=OllamaEmbeddings(model='mistral', temperature=0.9),
  persist_directory='./data'
)
vectordb.persist()
print('created vectorDB')


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=Ollama(model='mistral', temperature=0.9),
    retriever=vectordb.as_retriever(search_kwargs={'k': 7}),
    return_source_documents=True
)
print('created QA chain')

chat_hist = []

while True:
    #execute queries in qa chain
    userInp = input('[+] Write your query here: \n')
    if userInp == 'q':
        print('exiting...')
        break
    anws = qa_chain({'question': userInp, 'chat_history': chat_hist})
    print("[Mistral]" + anws['answer'])
    chat_hist.append((userInp, anws['answer']))
    print('\n\n')
# # we can now execute queries against our Q&A chain
# result = qa_chain.invoke({'query': 'Who is the CV about?'})
# print(result['result'])