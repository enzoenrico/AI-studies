import ollama

modelfile = """
FROM mistral 
SYSTEM You are a senior Engineer at a oil and gas company, everything that you awnser is based on your experience in the oil and gas industry. 
"""

print("Loading model...")
ollama.create(model="mistral", modelfile=modelfile)
while True:
    userInput = input('[+]Ask me anything: \n')
    res = ollama.chat(
        model="mistral",
        messages=[
            {"role": "user", "content": userInput}
        ],
        stream=True,

    )
    print("[Ollama]: ", end="", flush=True)
    for chunk in res:
        print(chunk["message"]["content"], end="", flush=True)
    print()
