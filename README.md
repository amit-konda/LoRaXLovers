# Customer Review RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for quickly searching and summarizing customer reviews from a text file.

## Features

- **Fast Search**: Semantic search through customer reviews using vector embeddings
- **Intelligent Summarization**: Generate summaries of reviews based on queries
- **Interactive Mode**: Command-line interface for easy querying
- **Efficient Storage**: FAISS vector store for fast similarity search
- **Flexible**: Works with local embeddings or OpenAI API

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. **Google Gemini API is already configured!** Your API key is set in `config.py`.

The default model is `gemini-2.5-flash`. You can change it in `config.py` or via environment variable:
```bash
export GEMINI_MODEL="gemini-2.5-pro"  # For more powerful model
```

## Usage

### Web Dashboard (Recommended)

Launch the interactive web dashboard:

```bash
streamlit run dashboard.py
```

This will open a web browser with a user-friendly interface where you can:
- 🔍 Search for reviews
- 📝 Generate AI summaries
- 📊 View review statistics

### Command-Line Usage

Run the interactive mode:

```bash
python main.py
```

This will:
1. Load the reviews from `Dell RAG compressed.txt`
2. Build a vector store (saved to `vectorstore/` directory)
3. Start an interactive session where you can search and summarize reviews

### Command-Line Options

```bash
python main.py [OPTIONS]
```

**Options:**
- `--data-file PATH`: Path to the reviews data file (default: "Dell RAG compressed.txt")
- `--vectorstore-path PATH`: Path to save/load the vector store (default: "vectorstore")
- `--rebuild`: Rebuild the vector store even if it exists
- `--query QUERY`: Search query (required for search/summarize modes)
- `--k NUMBER`: Number of results to return (default: 5)
- `--mode MODE`: Mode to run in - "search", "summarize", or "interactive" (default: "interactive")
- `--use-gemini`: Use Google Gemini for summarization (default: enabled)
- `--no-gemini`: Disable Gemini and use simple summarization
- `--gemini-api-key KEY`: Gemini API key (already set in config.py)
- `--gemini-model NAME`: Gemini model name (default: gemini-2.5-flash)

### Examples

**Search for reviews:**
```bash
python main.py --mode search --query "battery life issues" --k 10
```

**Summarize reviews:**
```bash
python main.py --mode summarize --query "overheating problems" --k 5
```

**Use Gemini for AI-powered summaries:**
```bash
# Using Gemini (default, API key from config.py)
python3 main.py --mode summarize --query "customer satisfaction" --k 5

# With a different Gemini model
python3 main.py --mode summarize --query "customer satisfaction" \
  --gemini-model "gemini-2.5-pro" --k 5

# Disable Gemini (use simple summarization)
python3 main.py --mode summarize --query "customer satisfaction" --no-gemini --k 5
```

**Rebuild the vector store:**
```bash
python main.py --rebuild
```

### Interactive Mode Commands

When in interactive mode, you can use:

- Type a query directly to search: `battery problems`
- Use `search:` prefix: `search: display issues`
- Use `summarize:` prefix: `summarize: overall customer satisfaction`
- Type `quit` or `exit` to exit

## How It Works

1. **Data Loading**: The pipeline parses customer reviews from the text file, extracting ratings, titles, and review bodies.

2. **Text Processing**: Reviews are cleaned (HTML tags removed, whitespace normalized) and split into chunks for efficient processing.

3. **Embedding**: Each review chunk is converted to a vector embedding using sentence transformers (default: `all-MiniLM-L6-v2`).

4. **Vector Store**: Embeddings are stored in a FAISS vector database for fast similarity search.

5. **Search**: When you query, the system finds the most similar review chunks using cosine similarity.

6. **Summarization**: Relevant reviews are combined and summarized, either using a simple algorithm or OpenAI's GPT models.

## Project Structure

```
.
├── dashboard.py            # Streamlit web dashboard
├── main.py                 # Main application script
├── data_loader.py          # Functions for loading and parsing reviews
├── rag_pipeline.py         # RAG pipeline implementation
├── rag_metrics.py          # RAG performance metrics and KPIs
├── acceptance_tests.py     # Acceptance tests for KPIs
├── config.py               # API key configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── KPI_DOCUMENTATION.md   # Detailed KPI explanations
├── Dell RAG compressed.txt # Customer reviews data file
└── vectorstore/           # Saved vector store (created after first run)
```

## Dependencies

- `langchain`: Framework for building LLM applications
- `faiss-cpu`: Vector similarity search library
- `sentence-transformers`: For generating embeddings
- `pandas`: Data manipulation
- `google-generativeai`: For Google Gemini API integration
- `streamlit`: Web dashboard framework

## Performance Metrics (KPIs)

The system tracks two key performance indicators:

1. **Retrieval Precision**: Measures how many retrieved documents are actually relevant
2. **Semantic Similarity Score**: Measures how semantically close retrieved docs are to the query

These KPIs are automatically calculated and displayed in the dashboard. See `KPI_DOCUMENTATION.md` for detailed explanations.

### Running Acceptance Tests

Test the KPI calculations with:
```bash
python3 acceptance_tests.py
```

This runs:
- **Happy Path Test**: Well-formed query with expected high KPIs
- **Edge Case Test**: Out-of-scope query with expected low KPIs
- **Empty Query Test**: System handles edge cases gracefully

## Notes

- The first run will take longer as it builds the vector store
- Subsequent runs will be faster as the vector store is loaded from disk
- The vector store is saved locally, so you don't need to rebuild it each time
- KPIs are calculated automatically for each search query
- See `KPI_DOCUMENTATION.md` for detailed KPI explanations and interpretation guides

## Troubleshooting

**Error: "Vector store not initialized"**
- Make sure you've run the pipeline at least once to build the vector store
- Or use `--rebuild` flag to rebuild it

**Error: "Data file not found"**
- Check that `Dell RAG compressed.txt` is in the current directory
- Or specify the correct path with `--data-file`

**OpenAI API errors**
- Make sure your API key is set correctly
- Check that you have API credits available
- The pipeline will fall back to simple summarization if OpenAI fails

