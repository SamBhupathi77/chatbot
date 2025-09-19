from typing import List, Dict, Any
import cohere
from config.settings import config

class CohereReranker:
    """Cohere Reranker implementation"""
    
    def __init__(self):
        self.client = cohere.Client(config.cohere_api_key)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: The search query
            documents: List of document dicts with 'text' and 'metadata' keys
            top_k: Number of top results to return (defaults to config.top_k)
            
        Returns:
            List of reranked documents with relevance scores
        """
        if not documents:
            return []
            
        top_k = top_k or config.top_k
        doc_texts = [doc["text"] for doc in documents]
        
        try:
            # Get reranked results from Cohere
            results = self.client.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(documents)),
                model="rerank-english-v2.0"
            )
            
            # Map scores back to original documents with metadata
            reranked_docs = []
            for result in results:
                doc = documents[result.index]
                reranked_docs.append({
                    **doc,
                    "relevance_score": result.relevance_score
                })
                
            return reranked_docs
            
        except Exception as e:
            print(f"Error in Cohere reranking: {str(e)}")
            # Return original documents if reranking fails
            return documents[:top_k]
