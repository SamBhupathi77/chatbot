import os
import json
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
import time

from config.settings import config
from src.models.embeddings import EmbeddingsAWSBedrock
from src.models.llm import ClaudeBedrockChatModel
from src.models.reranker import CohereReranker
from src.services.vector_store import SupabaseVectorStore
from src.services.session_manager import SessionManager, ChatSession

class AIAgent:
    """Main AI Agent that orchestrates all components with session management"""
    
    def __init__(self):
        self.session_manager = SessionManager(config.context_window_length)
        self.embeddings = EmbeddingsAWSBedrock()
        self.reranker = CohereReranker()
        self.vector_store = SupabaseVectorStore()
        self.chat_model = ClaudeBedrockChatModel()

    def create_new_chat(self, session_id: str = None) -> Dict[str, Any]:
        """Create a new chat session"""
        session = self.session_manager.create_session(session_id)
        return {"session_id": session.session_id}

    def delete_chat(self, session_id: str) -> bool:
        """Delete a specific chat session"""
        return self.session_manager.delete_session(session_id)

    def list_chats(self) -> List[Dict[str, Any]]:
        """List all active chat sessions"""
        return self.session_manager.list_sessions()

    def clear_chat_history(self, session_id: str) -> bool:
        """Clear history for a specific chat session"""
        session = self.session_manager.get_session(session_id)
        if session:
            session.clear()
            return True
        return False

    def process_message(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """Process user message through the RAG pipeline with session management"""
        # Get or create session
        session = self.session_manager.get_session(session_id)
        if not session:
            session = self.session_manager.create_session(session_id)
        
        # Add user message to session
        session.add_message("user", user_message)
        
        # Get conversation context
        context = session.get_context()
        
        # Generate response using the chat model
        response = self.chat_model.generate_response(context)
        
        # Add assistant response to session
        if "response" in response and response["response"]:
            session.add_message("assistant", response["response"])
        
        return {
            "session_id": session.session_id,
            "response": response.get("response", "I'm sorry, I couldn't generate a response."),
            "metadata": {
                "model": config.llm_model_id,
                "usage": response.get("usage", {})
            }
        }

# Create Flask application
app = Flask(__name__)
agent = AIAgent()

@app.route('/')
def health_check():
    return jsonify({"status": "ok", "message": "RAG Chatbot API is running"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        response = agent.process_message(user_message, session_id)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sessions', methods=['GET'])
def list_sessions():
    sessions = agent.list_chats()
    return jsonify({"sessions": sessions})

@app.route('/sessions', methods=['POST'])
def create_session():
    session_id = request.json.get('session_id')
    result = agent.create_new_chat(session_id)
    return jsonify(result)

@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    success = agent.delete_chat(session_id)
    if success:
        return jsonify({"status": "success", "message": f"Session {session_id} deleted"})
    return jsonify({"error": "Session not found"}), 404

def run_flask(port=5000):
    app.run(port=port)

def run_with_ngrok(port=5000):
    # Start ngrok when the app is run
    public_url = ngrok.connect(port).public_url
    print(f' * Public URL: {public_url}')
    run_flask(port)

if __name__ == '__main__':
    # Run with ngrok in a separate thread
    import threading
    thread = threading.Thread(target=run_with_ngrok, daemon=True)
    thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
