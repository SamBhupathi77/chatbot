import boto3
import os
from typing import List
from config import settings

class BedrockEmbedder:
    def __init__(self):
        """Initialize AWS Bedrock embeddings"""
        try:
            # Configure AWS credentials
            os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
            os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
            os.environ['AWS_DEFAULT_REGION'] = settings.AWS_REGION

            # Initialize Bedrock client
            self.client = boto3.client('bedrock-runtime', region_name=settings.AWS_REGION)
            self.model_id = settings.MODEL_ID
            print("✅ Successfully initialized Bedrock embeddings")
            
        except Exception as e:
            print(f"❌ Error initializing Bedrock client: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text query"""
        try:
            response = self.client.invoke_model(
                body=json.dumps({"inputText": text}),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(response.get('body').read())
            return response_body.get('embedding', [])
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings
