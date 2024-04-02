from langchain_community.document_loaders import PyPDFLoader

from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama

#load the doc for the model to read
pdfLoader = PyPDFLoader("Resume.pdf")
docs = pdfLoader.load()

#declare the used llm
model = Ollama(
    model="mistral",
    temperature=0.9,
)

chain = load_qa_chain(llm=model)
query = "What is your name?"
response = chain.invoke({"input_documents": docs, "question": query})
print(response['output_text'])