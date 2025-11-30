from dotenv import load_dotenv
import os
from google import genai

# Force load from absolute path
load_dotenv(r"D:\AI_INTERVIEW_AGENT\.env", override=True)

key = os.getenv("GEMINI_API_KEY")
print("KEY VALUE:", key)   # DEBUG FULL VALUE (TEMP)

client = genai.Client(api_key=key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hello like an HR interviewer in one sentence."
)

print(response.text)
