import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=APP_DIR / ".env")


class Config(BaseModel):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "gpt-4.1-mini"
    NOUGAT_MODEL: str = "facebook/nougat-base"
    TABLE_DET_MODEL: str = "microsoft/table-transformer-detection"


config = Config()
