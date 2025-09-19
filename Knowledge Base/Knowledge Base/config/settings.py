import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
SUPABASE_TABLE_NAME = os.getenv('SUPABASE_TABLE_NAME', 'qack')

# File Paths
LOCAL_FILE_PATH = os.getenv('LOCAL_FILE_PATH', 'data/input.json')
OUTPUT_JSON_PATH = 'data/output.json'

# Model Configuration
MODEL_ID = "amazon.titan-embed-text-v1"

# Text Processing
CHUNK_SIZE = 700
CHUNK_OVERLAP = 60
BATCH_SIZE = 20
