from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
for m in client.models.list():
    print(m.name, m.supported_actions)

    