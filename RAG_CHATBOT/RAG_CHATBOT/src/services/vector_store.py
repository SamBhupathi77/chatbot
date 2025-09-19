from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from config.settings import config
import json

class SupabaseVectorStore:
    """Supabase Vector Store implementation"""
    
    def __init__(self):
        self.client: Client = create_client(config.supabase_url, config.supabase_key)
        self.table_name = config.supabase_table

    def similarity_search(
        self, 
        embedding: List[float], 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            embedding: The query embedding vector
            top_k: Number of results to return (defaults to config.top_k)
            
        Returns:
            List of similar documents with metadata
        """
        top_k = top_k or config.top_k
        
        try:
            # Convert embedding to JSON string for Supabase
            embedding_str = json.dumps(embedding)
            
            # Query Supabase for similar vectors
            result = self.client.rpc(
                'match_documents',
                {
                    'query_embedding': embedding_str,
                    'match_count': top_k,
                    'filter': {'table_name': self.table_name}
                }
            ).execute()
            
            # Parse and return results
            return [
                {
                    "id": doc["id"],
                    "text": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "similarity": doc.get("similarity", 0.0)
                }
                for doc in result.data
            ]
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare documents for insertion
            records = []
            for doc in documents:
                record = {
                    "content": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "embedding": doc.get("embedding")
                }
                if not record["embedding"]:
                    continue
                records.append(record)
            
            # Insert into Supabase
            if records:
                self.client.table(self.table_name).upsert(records).execute()
                return True
            return False
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False
