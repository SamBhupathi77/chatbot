# Knowledge Base RAG Pipeline

A modular and organized implementation of a RAG (Retrieval-Augmented Generation) pipeline for processing and storing documents with vector embeddings in Supabase.

## Project Structure

```
.
├── config/
│   └── settings.py         # Configuration settings
├── data/                   # Input/output data directory
├── src/
│   ├── data_processing/    # Data loading and processing
│   │   └── processor.py
│   ├── embeddings/         # Embedding generation
│   │   └── bedrock_embeddings.py
│   ├── vector_store/       # Vector database operations
│   │   └── supabase_store.py
│   └── utils/              # Utility functions
│       └── text_splitter.py
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
└── .env.example          # Example environment variables
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd knowledge-base-rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Prepare your input data**
   - Place your input JSON file in the `data/` directory
   - Update `LOCAL_FILE_PATH` in `.env` if needed

## Usage

1. **Run the pipeline**
   ```bash
   python main.py
   ```

2. **Monitor the output**
   - The script will process the input file
   - Generate embeddings for the content
   - Store the results in your Supabase database

## Configuration

Edit `config/settings.py` or set environment variables in `.env` to configure:
- AWS credentials for Bedrock
- Supabase connection details
- File paths
- Chunking parameters
- Batch sizes

## Dependencies

- Python 3.8+
- AWS Bedrock access
- Supabase account
- Required Python packages (see requirements.txt)

## License

MIT
