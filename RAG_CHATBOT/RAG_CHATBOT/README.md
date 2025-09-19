# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot powered by AWS Bedrock (Claude 3.5 Sonnet) and Supabase vector store.

## Project Structure

```
RAG_CHATBOT/
├── .env.example           # Example environment variables
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── app.py                # Main application entry point
├── config/               # Configuration settings
│   ├── __init__.py
│   └── settings.py       # Application configuration
└── src/                  # Source code
    ├── __init__.py
    ├── models/           # AI models
    │   ├── __init__.py
    │   ├── embeddings.py # Embedding models
    │   ├── llm.py        # Language models
    │   └── reranker.py   # Reranking models
    └── services/         # Business logic
        ├── __init__.py
        ├── session_manager.py  # Chat session management
        └── vector_store.py     # Vector database operations
```

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Cohere API key
- Supabase project
- Python virtual environment (recommended)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG_CHATBOT
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and update with your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your actual API keys and configuration.

## Running the Application

### Development Server

```bash
python app.py
```

This will start the Flask development server on `http://localhost:5000`.

### With Ngrok (for public access)

```bash
python app.py --ngrok
```

This will start the server and create an ngrok tunnel for public access.

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Send a message to the chatbot
  ```json
  {
    "message": "Your message here",
    "session_id": "optional-session-id"
  }
  ```
- `GET /sessions` - List all active chat sessions
- `POST /sessions` - Create a new chat session
- `DELETE /sessions/<session_id>` - Delete a chat session

## Environment Variables

See `.env.example` for all available configuration options.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
