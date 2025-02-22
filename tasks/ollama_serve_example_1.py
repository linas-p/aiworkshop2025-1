import ollama

response = ollama.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': 'What is the capital of Lithuania?'
    },
])

print(response['message']['content'])