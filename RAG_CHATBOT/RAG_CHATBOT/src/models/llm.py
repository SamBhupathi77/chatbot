import json
from typing import Dict, List, Optional
import boto3
from config.settings import config

class ClaudeBedrockChatModel:
    """Claude 3.5 Sonnet Chat Model implementation using AWS Bedrock Converse API"""
    
    def __init__(self):
        self.client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_region
        )
        self.model_id = config.llm_model_id

    def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate response using AWS Bedrock Converse API"""
        try:
            # Convert messages to Bedrock format
            messages_formatted = self._convert_messages_to_bedrock_format(messages)
            
            # Call the Bedrock API
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages_formatted
            )
            
            # Process and return the response
            return {
                "response": response.get('output', {}).get('message', {}).get('content', []),
                "usage": response.get('usage', {})
            }
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {"response": "I'm sorry, I encountered an error while processing your request.", "error": str(e)}

    def _convert_messages_to_bedrock_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI-style messages to Bedrock Converse API format"""
        formatted_messages = []
        
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            formatted_messages.append({
                "role": role,
                "content": [{"text": msg["content"]}]
            })
            
        return formatted_messages
