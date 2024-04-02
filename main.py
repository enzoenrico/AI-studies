from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

model = Ollama(
    model="mistral",
    temperature=0.9,
)
print("Model loaded")

pdf = PyPDFLoader("Resume.pdf")
docs = pdf.load()

chain = load_qa_chain(llm=model, verbose=True)

query = input("Ask me anything: \n")

res = chain.invoke({"input_documents": docs, "question": query})

print(res['output_text'])
