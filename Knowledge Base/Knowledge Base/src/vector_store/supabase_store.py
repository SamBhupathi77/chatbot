from typing import List, Dict, Any
from supabase import create_client
from config import settings

class SupabaseVectorStore:
    """Custom vector store for Supabase integration"""
    
    def __init__(self):
        """Initialize Supabase client"""
        try:
            self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            self.table_name = settings.SUPABASE_TABLE_NAME
            print("✅ Successfully connected to Supabase")
        except Exception as e:
            print(f"❌ Error connecting to Supabase: {e}")
            raise

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]) -> bool:
        """Add documents with their embeddings to Supabase"""
        try:
            docs_to_insert = []
            
            for doc, embedding in zip(documents, embeddings):
                doc_data = {
                    'content': doc['page_content'],
                    'metadata': doc['metadata'],
                    'embedding': embedding
                }
                docs_to_insert.append(doc_data)
            
            # Insert in batches
            for i in range(0, len(docs_to_insert), settings.BATCH_SIZE):
                batch = docs_to_insert[i:i + settings.BATCH_SIZE]
                response = self.client.table(self.table_name).insert(batch).execute()
                
                if response.data:
                    print(f"✅ Successfully inserted batch of {len(batch)} documents")
                else:
                    print(f"⚠️ Warning: No data returned for batch {i//settings.BATCH_SIZE + 1}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error inserting documents: {e}")
            if hasattr(e, 'details'):
                print(f"   Details: {e.details}")
            if hasattr(e, 'hint'):
                print(f"   Hint: {e.hint}")
            return False
