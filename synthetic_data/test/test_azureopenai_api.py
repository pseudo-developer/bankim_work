import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Read env vars
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not endpoint:
    raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")

# Create client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-12-01-preview"
)

DEPLOYMENT_NAME = "gpt-4o-mini"  # <-- your deployment name

try:
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say exactly: Azure OpenAI API is working"}
        ],
        max_tokens=20
    )

    print("✅ API CALL SUCCESSFUL")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("❌ API CALL FAILED")
    print(e)
