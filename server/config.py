from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    # Use 'openai' or 'gemini'
    # Use 'hf' (Hugging Face local), 'gemini' (Google), or 'openai'
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'hf')
    # If using Google Generative API with an API key (less recommended than service account):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    # Path to Google service account JSON (optional alternative to API key)
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    # Local HF model name for `transformers` pipeline
    LOCAL_HF_MODEL = os.getenv('LOCAL_HF_MODEL', 'gpt2')
    MODEL_MODE = os.getenv('MODEL_MODE', 'llm')


settings = Settings()
