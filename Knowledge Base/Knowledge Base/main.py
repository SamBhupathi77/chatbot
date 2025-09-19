import os
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from config import settings
from src.data_processing.processor import DataProcessor
from src.embeddings.bedrock_embeddings import BedrockEmbedder
from src.vector_store.supabase_store import SupabaseVectorStore
from src.utils.text_splitter import TextSplitter

class RAGPipeline:
    def __init__(self):
        self.processor = DataProcessor()
        self.embedder = BedrockEmbedder()
        self.vector_store = SupabaseVectorStore()
        self.text_splitter = TextSplitter()

    def process_file(self, file_path: str) -> List[Dict]:
        """Process input file and return documents"""
        # Load and process data
        json_data = self.processor.extract_json_data(file_path)
        if not json_data:
            raise ValueError("Failed to load JSON data")
        
        processed_data = self.processor.process_json_data(json_data)
        if not processed_data:
            raise ValueError("No valid data found in the input file")
        
        # Create documents
        documents = self.processor.create_documents(processed_data)
        if not documents:
            raise ValueError("No valid documents could be created")
        
        return documents

    def run(self, input_file: str):
        """Run the complete RAG pipeline"""
        print("🚀 Starting RAG Pipeline...")
        
        try:
            # 1. Process input file
            print("📄 Processing input file...")
            documents = self.process_file(input_file)
            
            # 2. Split documents into chunks
            print("✂️  Splitting documents into chunks...")
            split_docs = self.text_splitter.split_documents(documents)
            print(f"✅ Split into {len(split_docs)} chunks")
            
            # 3. Generate embeddings
            print("🤖 Generating embeddings...")
            texts = [doc['page_content'] for doc in split_docs]
            embeddings = self.embedder.embed_documents(texts)
            
            # 4. Store in vector database
            print("💾 Storing in vector database...")
            success = self.vector_store.add_documents(split_docs, embeddings)
            
            if success:
                print("\n🎉 RAG Pipeline completed successfully!")
            else:
                print("\n❌ RAG Pipeline completed with errors")
                
        except Exception as e:
            print(f"\n❌ Error in RAG Pipeline: {str(e)}")
            raise

def main():
    # Ensure input file exists
    input_file = settings.LOCAL_FILE_PATH
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please create a .env file with the correct LOCAL_FILE_PATH or set it in config/settings.py")
        return
    
    # Run the pipeline
    pipeline = RAGPipeline()
    pipeline.run(input_file)

if __name__ == "__main__":
    main()
