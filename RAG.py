from llama_index.llms.ollama import Ollama
import requests


llm = Ollama(model="mistral", request_timeout=600)

res = llm.complete('do you know about claude-3?')
print(res.text)