from typing import List, Dict, Any
from config import settings

class TextSplitter:
    """Handles text splitting for documents"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []
            
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start position forward, accounting for overlap
            start = end - self.chunk_overlap
            if start <= 0:  # Prevent infinite loop with very short texts
                break
                
        return chunks
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split multiple documents into chunks"""
        split_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc['page_content'])
            
            for i, chunk in enumerate(chunks):
                new_doc = doc.copy()
                new_doc['page_content'] = chunk
                new_doc['metadata'] = new_doc.get('metadata', {}).copy()
                new_doc['metadata']['chunk'] = i + 1
                split_docs.append(new_doc)
        
        return split_docs
