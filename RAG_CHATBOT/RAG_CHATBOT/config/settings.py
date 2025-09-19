import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    # AWS Configuration
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Cohere Configuration
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    
    # Supabase Configuration
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    supabase_table: str = os.getenv("SUPABASE_TABLE", "zack")
    
    # Application Configuration
    webhook_id: str = os.getenv("WEBHOOK_ID", "")
    context_window_length: int = int(os.getenv("CONTEXT_WINDOW_LENGTH", "3"))
    top_k: int = int(os.getenv("TOP_K_RESULTS", "15"))
    
    # Model Configuration
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")

# Create a single config instance to be imported throughout the application
config = Config()
