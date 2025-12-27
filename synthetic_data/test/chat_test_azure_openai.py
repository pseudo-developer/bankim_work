import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([API_KEY, ENDPOINT, DEPLOYMENT_NAME, API_VERSION]):
    raise ValueError("One or more Azure OpenAI env variables are missing")

# Create client
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION
)

print("✅ Azure OpenAI connected")
print("Type your question (type 'exit' to quit)\n")

# Simple interactive loop
while True:
    user_input = input("You: ")

    if user_input.lower() in {"exit", "quit"}:
        print("👋 Exiting chat")
        break

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200
        )

        print("AI:", response.choices[0].message.content.strip(), "\n")

    except Exception as e:
        print("❌ Error:", e)
        break
