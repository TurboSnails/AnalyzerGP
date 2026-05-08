import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db"