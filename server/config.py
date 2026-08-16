from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '39f6bffedb5f463bab17fe870dbd0ec5')
    # Provider: 'hf' (Hugging Face local/API pipeline)
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'hf')
    # Local HF model name for transformers pipeline
    LOCAL_HF_MODEL = os.getenv('LOCAL_HF_MODEL', 'gpt2')
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    MODEL_MODE = os.getenv('MODEL_MODE', 'local')


settings = Settings()
