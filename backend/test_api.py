import requests

url = "http://127.0.0.1:5000/chat"
data = {"message": "Hello Gemini, how are you?"}

response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Response:", response.json())
