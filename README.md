
1. Clone the Repository

git clone <your-repo-url>
cd flight-booking-rag


2. Create Virtual Environment

python -m venv llm_env

llm_env\Scripts\activate


3. Install Dependencies

pip install -r requirements.txt


4. Set up Ollama and Gemma

Start Ollama server: ollama serve


1. Initialize ChromaDB with Data

python process_flight_data.py

This loads your flight booking Q&A data into ChromaDB for semantic search.

2. Test the Retriever - python retriever.py

This tests document retrieval functionality with sample queries.

3. Run the Complete RAG Pipeline - python rag_with_gemma.py (change to gemma3:2b or whatever local llm is available in this script)

This demonstrates the full pipeline: query → retrieve → generate response.


## Configuration

### ChromaDB Settings
- **Storage Path**: `./chroma_db`
- **Collection Name**: `flight_info`
- **Embedding Model**: `all-MiniLM-L6-v2`

### Gemma API Settings
- **Endpoint**: `http://127.0.0.1:11434/v1/completions`
- **Model**: `gemma3:1b`
- **Temperature**: 0.7
- **Top-p**: 0.95


1. ChromaDB Collection Exists Error
   - Delete existing collection or use different name
   - Run `process_flight_data.py` with option to recreate

2. Ollama Connection Error
   - Ensure Ollama is running: `ollama serve`
   - Check if Gemma model is installed: `ollama list`

3. Import Errors
   - Ensure virtual environment is activated
   - Install missing packages: `pip install -r requirements.txt`

