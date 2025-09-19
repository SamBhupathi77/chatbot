!pip install flask==2.3.3 openai==0.28.1  numpy==1.24.3 requests==2.31.0 python-dotenv==1.0.0 gunicorn==21.2.0

!pip install cohere

!pip install boto3

!pip install supabase==2.0.0

!pip install pyngrok

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
import requests
import openai
import cohere
import boto3
from supabase import create_client, Client
from flask import Flask, request, jsonify
import uuid
import threading
from pyngrok import ngrok
import time

# Set environment variables directly in Colab
os.environ['OPENAI_API_KEY'] = ''
os.environ['COHERE_API_KEY'] = ''
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = '//'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['SUPABASE_URL'] = ''
os.environ['SUPABASE_KEY'] = ''
os.environ['SUPABASE_TABLE'] = ''
os.environ['WEBHOOK_ID'] = ''

import json
import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

import boto3
import cohere
from flask import Flask, request, jsonify
from supabase import create_client, Client

try:
    from pyngrok import ngrok
except ImportError:
    print("Warning: pyngrok not installed. Install with: pip install pyngrok")
    ngrok = None

@dataclass
class Config:
    cohere_api_key: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    supabase_url: str
    supabase_key: str
    supabase_table: str = "zack"
    context_window_length: int = 3
    top_k: int = 15
    webhook_id: str = ""

# System prompt from your workflow
SYSTEM_MESSAGE = """You are a Motive customer support agent focused on delivering accurate and maximally concise responses.
PRIMARY OBJECTIVES:
Provide complete, factually correct answers (Correctness)
Deliver minimal viable information that fully answers the question (Conciseness)
RESPONSE GUIDELINES:
Accuracy First: Answer must be complete and precise based on available information

Include all relevant details that directly answer the question
If information is incomplete, state what is known rather than guessing
Use specific numbers, percentages, and technical details when available
Maximum Conciseness Rules:

Lead with the direct answer in the first sentence
Answer ONLY what was asked - no additional context unless essential
Remove ALL unnecessary descriptors, titles, and background information
Use bullet points for lists instead of sentences
Eliminate all filler phrases, redundant explanations, or summaries
Every single word must directly contribute to answering the question
Structure:

Start with the exact answer requested
Stop immediately when the question is fully answered
Use active voice and direct language
No conversational elements or closing statements
TONE: Direct and factual. Prioritize information density over any social pleasantries.
CORE RULE: If the question asks for X, provide only X. Nothing more, nothing less."""

class ChatSession:
    """Individual chat session with its own memory"""
    def __init__(self, session_id: str, window_length: int = 3):
        self.session_id = session_id
        self.window_length = window_length
        self.messages = deque(maxlen=window_length * 2)  # user + assistant pairs
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

    def add_message(self, role: str, content: str):
        """Add a message to the session memory"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()

    def get_context(self) -> List[Dict[str, str]]:
        """Get conversation context for this session"""
        # Return only role and content for Bedrock Converse API compatibility
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]

    def clear(self):
        """Clear the session memory"""
        self.messages.clear()
        self.last_activity = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "window_length": self.window_length
        }

class SessionManager:
    """Manages multiple chat sessions"""
    def __init__(self, default_window_length: int = 3):
        self.sessions: Dict[str, ChatSession] = {}
        self.default_window_length = default_window_length

    def create_session(self, session_id: str = None) -> str:
        """Create a new chat session"""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id in self.sessions:
            # Clear existing session instead of creating duplicate
            self.sessions[session_id].clear()
        else:
            self.sessions[session_id] = ChatSession(session_id, self.default_window_length)

        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a specific session"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [session.get_summary() for session in self.sessions.values()]

    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Clean up sessions older than max_age_hours"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        inactive_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.last_activity < cutoff_time
        ]

        for session_id in inactive_sessions:
            del self.sessions[session_id]

        return len(inactive_sessions)

class EmbeddingsAWSBedrock:
    """AWS Bedrock Embeddings using Titan"""
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str):
        self.client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        self.model_id = "amazon.titan-embed-text-v1"

    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query text"""
        body = json.dumps({"inputText": text})

        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')

class CohereReranker:
    """Cohere Reranker implementation"""
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 15) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance"""
        if not documents:
            return []

        # Extract text content for reranking
        texts = [doc.get('content', '') for doc in documents]

        response = self.client.rerank(
            model='rerank-v3.5',
            query=query,
            documents=texts,
            top_n=min(top_k, len(texts))
        )

        # Return reranked documents
        reranked_docs = []
        for result in response.results:
            original_doc = documents[result.index]
            original_doc['relevance_score'] = result.relevance_score
            reranked_docs.append(original_doc)

        return reranked_docs

class SupabaseVectorStore:
    """Supabase Vector Store implementation"""
    def __init__(self, url: str, key: str, table_name: str):
        self.client: Client = create_client(url, key)
        self.table_name = table_name

    def similarity_search(self, embedding: List[float], top_k: int = 15) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        try:
            response = self.client.rpc(
                'match_documents',
                {
                    'query_embedding': embedding,
                    'match_count': top_k
                }
            ).execute()

            return response.data if response.data else []
        except Exception as e:
            print(f"Error in similarity search: {e}")
            try:
                response = self.client.rpc(
                    'match_documents',
                    {
                        'query_embedding': embedding,
                        'filter': '{}',
                        'match_count': top_k
                    }
                ).execute()

                return response.data if response.data else []
            except Exception as e2:
                print(f"Error with alternative function: {e2}")
                try:
                    response = self.client.table(self.table_name).select("*").limit(top_k).execute()
                    return response.data if response.data else []
                except Exception as e3:
                    print(f"Error with fallback query: {e3}")
                    return []

class ClaudeBedrockChatModel:
    """Claude 3.5 Sonnet Chat Model implementation using AWS Bedrock Converse API"""
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str, model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
        self.client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        self.model_id = model_id

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using AWS Bedrock Converse API"""
        try:
            # Convert messages to Bedrock Converse API format
            bedrock_messages = self._convert_messages_to_bedrock_format(messages)

            # Extract system message if present
            system_message = None
            if bedrock_messages and bedrock_messages[0]['role'] == 'system':
                system_message = bedrock_messages.pop(0)['content'][0]['text']

            # Prepare the request
            request_params = {
                'modelId': self.model_id,
                'messages': bedrock_messages,
                'inferenceConfig': {
                    'temperature': 0.1,
                    'maxTokens': 1000
                }
            }

            # Add system message if present
            if system_message:
                request_params['system'] = [{'text': system_message}]

            # Make the API call
            response = self.client.converse(**request_params)

            # Extract the response text
            if 'output' in response and 'message' in response['output']:
                content = response['output']['message']['content']
                if content and len(content) > 0:
                    return content[0]['text']

            return "Error: No response content received from Claude"

        except Exception as e:
            return f"Error generating response: {e}"

    def _convert_messages_to_bedrock_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Bedrock Converse API format"""
        bedrock_messages = []

        for message in messages:
            role = message['role']
            content = message['content']

            # Handle system messages
            if role == 'system':
                bedrock_messages.append({
                    'role': 'system',
                    'content': [{'text': content}]
                })
            # Handle user messages
            elif role == 'user':
                bedrock_messages.append({
                    'role': 'user',
                    'content': [{'text': content}]
                })
            # Handle assistant messages
            elif role == 'assistant':
                bedrock_messages.append({
                    'role': 'assistant',
                    'content': [{'text': content}]
                })

        return bedrock_messages

class AIAgent:
    """Main AI Agent that orchestrates all components with session management"""
    def __init__(self, config: Config):
        self.config = config
        self.session_manager = SessionManager(config.context_window_length)
        self.embeddings = EmbeddingsAWSBedrock(
            config.aws_access_key_id,
            config.aws_secret_access_key,
            config.aws_region
        )
        self.reranker = CohereReranker(config.cohere_api_key)
        self.vector_store = SupabaseVectorStore(
            config.supabase_url,
            config.supabase_key,
            config.supabase_table
        )
        self.chat_model = ClaudeBedrockChatModel(
            config.aws_access_key_id,
            config.aws_secret_access_key,
            config.aws_region
        )

    def create_new_chat(self, session_id: str = None) -> str:
        """Create a new chat session"""
        return self.session_manager.create_session(session_id)

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

    async def process_message(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """Process user message through the RAG pipeline with session management"""
        try:
            # Create default session if none provided
            if session_id is None:
                session_id = self.create_new_chat()

            # Get or create session
            session = self.session_manager.get_session(session_id)
            if session is None:
                session_id = self.create_new_chat(session_id)
                session = self.session_manager.get_session(session_id)

            # Step 1: Generate embeddings for the user query
            query_embedding = self.embeddings.embed_query(user_message)

            # Step 2: Retrieve similar documents from vector store
            similar_docs = self.vector_store.similarity_search(query_embedding, self.config.top_k)

            # Step 3: Rerank documents using Cohere
            if similar_docs:
                reranked_docs = self.reranker.rerank(user_message, similar_docs, self.config.top_k)
            else:
                reranked_docs = []

            # Step 4: Prepare context from retrieved documents
            context = ""
            if reranked_docs:
                context = "\n\n".join([
                    f"Document {i+1}: {doc.get('content', '')}"
                    for i, doc in enumerate(reranked_docs[:5])
                ])

            # Step 5: Prepare conversation history from session
            conversation_history = session.get_context()

            # Step 6: Build messages for Claude
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

            # Add conversation history
            messages.extend(conversation_history)

            # Add current user message with context
            user_message_with_context = user_message
            if context:
                user_message_with_context = f"Context from Motive knowledge base:\n{context}\n\nUser question: {user_message}"

            messages.append({"role": "user", "content": user_message_with_context})

            # Step 7: Generate response
            assistant_response = self.chat_model.generate_response(messages)

            # Step 8: Update session memory
            session.add_message("user", user_message)
            session.add_message("assistant", assistant_response)

            return {
                "response": assistant_response,
                "session_id": session_id,
                "message_count": len(session.messages),
                "context_used": len(reranked_docs) > 0
            }

        except Exception as e:
            return {
                "response": f"Error processing message: {e}",
                "session_id": session_id,
                "error": True
            }

    def chat(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """Synchronous chat method for easy usage"""
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.process_message(user_message, session_id))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.process_message(user_message, session_id))
            finally:
                loop.close()
        except ImportError:
            return self.process_message_sync(user_message, session_id)

    def process_message_sync(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """Synchronous version of process_message"""
        try:
            # Create default session if none provided
            if session_id is None:
                session_id = self.create_new_chat()

            # Get or create session
            session = self.session_manager.get_session(session_id)
            if session is None:
                session_id = self.create_new_chat(session_id)
                session = self.session_manager.get_session(session_id)

            # Step 1: Generate embeddings for the user query
            query_embedding = self.embeddings.embed_query(user_message)

            # Step 2: Retrieve similar documents from vector store
            similar_docs = self.vector_store.similarity_search(query_embedding, self.config.top_k)

            # Step 3: Rerank documents using Cohere
            if similar_docs:
                reranked_docs = self.reranker.rerank(user_message, similar_docs, self.config.top_k)
            else:
                reranked_docs = []

            # Step 4: Prepare context from retrieved documents
            context = ""
            if reranked_docs:
                context = "\n\n".join([
                    f"Document {i+1}: {doc.get('content', '')}"
                    for i, doc in enumerate(reranked_docs[:5])
                ])

            # Step 5: Prepare conversation history from session
            conversation_history = session.get_context()

            # Step 6: Build messages for Claude
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

            # Add conversation history
            messages.extend(conversation_history)

            # Add current user message with context
            user_message_with_context = user_message
            if context:
                user_message_with_context = f"Context from Motive knowledge base:\n{context}\n\nUser question: {user_message}"

            messages.append({"role": "user", "content": user_message_with_context})

            # Step 7: Generate response
            assistant_response = self.chat_model.generate_response(messages)

            # Step 8: Update session memory
            session.add_message("user", user_message)
            session.add_message("assistant", assistant_response)

            return {
                "response": assistant_response,
                "session_id": session_id,
                "message_count": len(session.messages),
                "context_used": len(reranked_docs) > 0
            }

        except Exception as e:
            return {
                "response": f"Error processing message: {e}",
                "session_id": session_id,
                "error": True
            }

class ChatTrigger:
    """Flask-based chat trigger with session management"""
    def __init__(self, config: Config):
        self.app = Flask(__name__)
        self.agent = AIAgent(config)
        self.webhook_id = config.webhook_id

        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/webhook/<webhook_id>', methods=['POST'])
        def chat_webhook(webhook_id):
            if webhook_id != self.webhook_id:
                return jsonify({"error": "Invalid webhook ID"}), 401

            try:
                data = request.get_json()
                user_message = data.get('message', '')
                session_id = data.get('session_id')  # Optional session ID

                if not user_message:
                    return jsonify({"error": "No message provided"}), 400

                # Process message through AI agent
                result = self.agent.process_message_sync(user_message, session_id)

                return jsonify({
                    "response": result["response"],
                    "session_id": result["session_id"],
                    "message_count": result.get("message_count", 0),
                    "context_used": result.get("context_used", False),
                    "status": "success"
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/new_chat', methods=['POST'])
        def new_chat():
            """Create a new chat session"""
            try:
                data = request.get_json() or {}
                session_id = data.get('session_id')
                new_session_id = self.agent.create_new_chat(session_id)
                return jsonify({
                    "session_id": new_session_id,
                    "message": "New chat session created",
                    "status": "success"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/list_chats', methods=['GET'])
        def list_chats():
            """List all active chat sessions"""
            try:
                sessions = self.agent.list_chats()
                return jsonify({
                    "sessions": sessions,
                    "total_count": len(sessions),
                    "status": "success"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/delete_chat/<session_id>', methods=['DELETE'])
        def delete_chat(session_id):
            """Delete a specific chat session"""
            try:
                success = self.agent.delete_chat(session_id)
                if success:
                    return jsonify({
                        "message": f"Chat session {session_id} deleted",
                        "status": "success"
                    })
                else:
                    return jsonify({"error": "Session not found"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/clear_chat/<session_id>', methods=['POST'])
        def clear_chat(session_id):
            """Clear chat history for a specific session"""
            try:
                success = self.agent.clear_chat_history(session_id)
                if success:
                    return jsonify({
                        "message": f"Chat history for session {session_id} cleared",
                        "status": "success"
                    })
                else:
                    return jsonify({"error": "Session not found"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy"})

    def run_with_ngrok(self, port=5000):
        """Run Flask app with ngrok tunnel for Colab"""
        if ngrok is None:
            print("Warning: ngrok not available. Install with: pip install pyngrok")
            return None

        public_url = ngrok.connect(port)
        print(f"\n🚀 Flask app is running with Claude 3.5 Sonnet!")
        print(f"📡 Public URL: {public_url}")
        print(f"🎯 Chat endpoint: {public_url}/webhook/{self.webhook_id}")
        print(f"🆕 New chat: {public_url}/new_chat")
        print(f"📋 List chats: {public_url}/list_chats")
        print(f"❌ Delete chat: {public_url}/delete_chat/<session_id>")
        print(f"🧹 Clear chat: {public_url}/clear_chat/<session_id>")
        print(f"❤️  Health check: {public_url}/health")

        def run_flask():
            self.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()

        return public_url

def create_app():
    """Create and configure the application"""
    config = Config(
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_region=os.getenv('AWS_REGION', 'us-east-1'),
        supabase_url=os.getenv('SUPABASE_URL'),
        supabase_key=os.getenv('SUPABASE_KEY'),
        supabase_table=os.getenv('SUPABASE_TABLE', 'zack')
    )

    required_vars = [
        'cohere_api_key', 'aws_access_key_id',
        'aws_secret_access_key', 'supabase_url', 'supabase_key'
    ]

    missing_vars = [var for var in required_vars if not getattr(config, var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    return ChatTrigger(config)

def start_chatbot_server():
    """Start the chatbot server with ngrok tunnel"""
    app = create_app()
    print("🤖 Starting RAG Chatbot Server with Claude 3.5 Sonnet and Session Management...")
    public_url = app.run_with_ngrok()
    return app, public_url

def create_simple_chatbot():
    """Create a simple chatbot instance for direct chat"""
    config = Config(
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_region=os.getenv('AWS_REGION', 'us-east-1'),
        supabase_url=os.getenv('SUPABASE_URL'),
        supabase_key=os.getenv('SUPABASE_KEY'),
        supabase_table=os.getenv('SUPABASE_TABLE', 'zack')
    )

    return AIAgent(config)

    bot = create_simple_chatbot()

    session_id = bot.create_new_chat()

    response = bot.chat("What OEMs does Motive support for reefer monitoring?",session_id)
print(f"Response: {response['response']}")