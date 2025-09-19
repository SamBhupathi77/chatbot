import boto3
from typing import List, Optional
from dataclasses import dataclass
from config.settings import config

@dataclass
class EmbeddingsAWSBedrock:
    """AWS Bedrock Embeddings using Titan"""
    
    def __init__(self):
        self.client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_region
        )
        self.model_id = config.embedding_model_id

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for a query text"""
        try:
            # Format the input for the Bedrock model
            body = json.dumps({
                "inputText": text
            })
            
            # Call the Bedrock API
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            return response_body.get('embedding')
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return None
