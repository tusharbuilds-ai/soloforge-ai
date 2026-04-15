from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

print(f"Key being used: {os.getenv('GOOGLE_API_KEY')}")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7
)