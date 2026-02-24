import os
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

class Config(BaseModel):
    OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY")
    CHUNK_SIZE:int = 1000
    CHUNK_OVERLAP:int = 200
    EMBEDDING_MODEL:int = "text-embedding-ada-002"
    
    LLM_MODEL:int = "gpt-4.1-mini"
    NOUGAT_MODEL:str = "facebook/nougat-base"
    TABLE_DET_MODEL:str = "microsoft/table-transformer-detection"