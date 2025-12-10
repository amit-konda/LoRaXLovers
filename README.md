# Customer Review RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for quickly searching and summarizing customer reviews from a text file, powered by local language models with style-based steering vectors.

## Features

- **Fast Search**: Semantic search through customer reviews using vector embeddings
- **AI-Powered Summarization**: Generate intelligent summaries using local language models (TinyLlama)
- **Style Control with Steering Vectors**: Control summary style (formal, casual, concise, detailed) using advanced steering vector techniques
- **Interactive Dashboard**: Beautiful web interface with real-time controls
- **Interactive CLI**: Command-line interface for easy querying
- **Efficient Storage**: FAISS vector store for fast similarity search
- **CPU Optimized**: Works efficiently on CPU with optional quantization

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. **Model Configuration**: The default model is TinyLlama (no authentication required). You can change it in `config.py` if needed:

```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Default (no auth needed)
```

### First Run Setup

On the first run, the system will automatically:

1. **Download Embedding Model**: The `all-MiniLM-L6-v2` embedding model (~90MB) will be downloaded automatically
2. **Build Vector Store**: The system will process `Dell RAG compressed.txt` and build a vector store (saved to `vectorstore/` directory)
3. **Download Language Model**: When you first generate a summary, TinyLlama (~2.3GB) will be downloaded automatically
4. **Train Steering Vectors**: When you first use a style (formal, casual, concise, detailed), the steering vector will be trained and cached

**Important Notes:**
- **Internet Required**: First run requires internet connection to download models
- **Disk Space**: Ensure you have at least 3-4GB free space for models
- **Time**: First run may take 5-10 minutes depending on internet speed
- **Caching**: All models and vectors are cached locally, so subsequent runs are much faster
- **No Authentication**: TinyLlama doesn't require Hugging Face authentication

## Usage

### Web Dashboard (Recommended)

Launch the interactive web dashboard:

```bash
streamlit run dashboard.py
```

Or use the convenience script:

```bash
./run_dashboard.sh
```

This will open a web browser with a user-friendly interface where you can:
- 🔍 Search for reviews
- 📝 Generate AI summaries with style control
- 🎨 Adjust steering vectors (style and strength)
- 📊 View review statistics
- 🧪 Run acceptance tests

**Dashboard Features:**
- **Style Selector**: Choose from balanced, formal, casual, concise, or detailed summaries
- **Steering Strength Control**: Adjust how strongly the style is applied (0.0 to 1.0)
- **Model Selection**: Configure TinyLlama model settings
- **Real-time Status**: See model loading status and device information
- **Steering Vector KPIs**: View Style Adherence and Content Quality scores for each summary
- **Performance Metrics**: Track retrieval and steering vector performance over time

**What Loads Automatically:**
- ✅ **Vector Store**: Automatically built from `Dell RAG compressed.txt` on first run
- ✅ **Embedding Model**: Downloads automatically when pipeline initializes
- ✅ **Language Model**: Downloads automatically when you first generate a summary
- ✅ **Steering Vectors**: Trained automatically on first use of each style, then cached to disk
- ✅ **All caches persist**: Everything is saved locally for instant subsequent loads

### Command-Line Usage

Run the interactive mode:

```bash
python3 main.py
```

This will:
1. Load the reviews from `Dell RAG compressed.txt`
2. Build a vector store (saved to `vectorstore/` directory)
3. Start an interactive session where you can search and summarize reviews

### Command-Line Options

```bash
python3 main.py [OPTIONS]
```

**Options:**
- `--data-file PATH`: Path to the reviews data file (default: "Dell RAG compressed.txt")
- `--vectorstore-path PATH`: Path to save/load the vector store (default: "vectorstore")
- `--rebuild`: Rebuild the vector store even if it exists
- `--query QUERY`: Search query (required for search/summarize modes)
- `--k NUMBER`: Number of results to return (default: 5)
- `--mode MODE`: Mode to run in - "search", "summarize", or "interactive" (default: "interactive")
- `--model MODEL`: Model name override (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
- `--device DEVICE`: Device to use - "auto", "cpu", or "cuda" (default: auto-detect)
- `--quantize`: Enable 8-bit quantization (recommended for CPU)
- `--no-quantize`: Disable quantization
- `--steering-strength FLOAT`: Steering vector strength (0.0 to 1.0, default: 0.5)
- `--style STYLE`: Summary style - "balanced", "formal", "casual", "concise", or "detailed" (default: balanced)
- `--hf-token TOKEN`: Hugging Face token for model access

### Examples

**Search for reviews:**
```bash
python3 main.py --mode search --query "battery life issues" --k 10
```

**Summarize reviews with default style:**
```bash
python3 main.py --mode summarize --query "overheating problems" --k 5
```

**Summarize with specific style:**
```bash
# Formal style
python3 main.py --mode summarize --query "customer satisfaction" --style formal --k 5

# Casual style
python3 main.py --mode summarize --query "product quality" --style casual --k 5

# Concise style
python3 main.py --mode summarize --query "common issues" --style concise --k 5
```

**Use different model (if configured):**
```bash
# Use a different model (must be compatible)
python3 main.py --mode summarize --query "customer satisfaction" \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --k 5
```

**Enable quantization for CPU:**
```bash
python3 main.py --mode summarize --query "customer satisfaction" \
  --quantize --k 5
```

**Rebuild the vector store:**
```bash
python3 main.py --rebuild
```

### Interactive Mode Commands

When in interactive mode, you can use:

- Type a query directly to search: `battery problems`
- Use `search:` prefix: `search: display issues`
- Use `summarize:` prefix: `summarize: overall customer satisfaction`
- Use `summarize:` with style: `summarize: customer satisfaction --style formal`
- Type `quit` or `exit` to exit

## How It Works

1. **Data Loading**: The pipeline parses customer reviews from the text file, extracting ratings, titles, and review bodies.

2. **Text Processing**: Reviews are cleaned (HTML tags removed, whitespace normalized) and split into chunks for efficient processing.

3. **Embedding**: Each review chunk is converted to a vector embedding using sentence transformers (default: `all-MiniLM-L6-v2`).

4. **Vector Store**: Embeddings are stored in a FAISS vector database for fast similarity search.

5. **Search**: When you query, the system finds the most similar review chunks using cosine similarity.

6. **Summarization with Steering Vectors**: 
   - Relevant reviews are retrieved and combined
   - A local language model (TinyLlama) generates the summary
   - **Steering vectors** are applied to control the style:
     - **Formal**: Professional, formal language
     - **Casual**: Conversational, friendly tone
     - **Concise**: Brief and to the point
     - **Detailed**: Comprehensive with explanations
     - **Balanced**: Standard, neutral summary
   - **Steering Vector KPIs** are calculated to measure:
     - **Style Adherence**: How well the output matches the intended style
     - **Content Quality**: How well the content maintains coherence and relevance

## Steering Vectors

Steering vectors are a technique that modifies the model's internal activations during generation to control output style without retraining. The system includes:

- **Pre-trained style vectors**: For formal, casual, concise, and detailed styles
- **Adjustable strength**: Control how strongly the style is applied (0.0 to 1.0)
- **Layer-specific application**: Applied at optimal transformer layers for best results
- **Real-time control**: Change style on-the-fly in the dashboard or CLI

## Project Structure

```
.
├── dashboard.py            # Streamlit web dashboard
├── main.py                 # Main application script
├── data_loader.py          # Functions for loading and parsing reviews
├── rag_pipeline.py         # RAG pipeline implementation with steering vectors
├── rag_metrics.py          # RAG performance metrics and KPIs
├── acceptance_tests.py     # Acceptance tests for KPIs
├── config.py               # Model and steering configuration
├── requirements.txt        # Python dependencies
├── run_dashboard.sh        # Convenience script to start dashboard
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
- `torch`: PyTorch for model inference
- `transformers`: Hugging Face transformers library
- `accelerate`: Efficient model loading
- `steering-vectors`: Steering vector implementation
- `optimum` & `bitsandbytes`: Quantization support
- `streamlit`: Web dashboard framework

## Model Information

### TinyLlama 1.1B (Default)
- **Pros**: Fast, no authentication required, works on CPU
- **Memory**: ~0.7GB with quantization, ~2.3GB without
- **Speed**: Very fast inference on CPU
- **Quality**: Good for most use cases
- **Setup**: No setup required, works immediately
- **License**: Apache 2.0 (open source)

## Performance Metrics (KPIs)

The system tracks multiple key performance indicators:

### Retrieval KPIs
1. **Retrieval Precision**: Measures how many retrieved documents are actually relevant
2. **Semantic Similarity Score**: Measures how semantically close retrieved docs are to the query

### Steering Vector KPIs
3. **Style Adherence Score**: Measures how well generated output matches the intended style (formal, casual, concise, detailed)
4. **Content Quality Preservation**: Ensures steering vectors don't degrade factual accuracy, coherence, or relevance

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

### Automatic Loading & Caching

The system automatically handles loading and caching:

- **Vector Store**: Automatically built on first run, saved to `vectorstore/` directory
- **Embedding Model**: Downloaded automatically, cached by sentence-transformers library
- **Language Model**: Downloaded on first summary generation, cached by Hugging Face
- **Steering Vectors**: Trained on first use of each style, cached to `steering_vectors_cache/` directory
- **All Caches Persist**: Everything is saved locally, so subsequent runs are instant

### Performance

- **First Run**: 5-10 minutes (downloads models, builds vector store)
- **Subsequent Runs**: < 10 seconds (everything loads from cache)
- **First Summary**: 30-60 seconds (downloads language model if not cached)
- **Subsequent Summaries**: 5-15 seconds (model and vectors loaded from cache)

### KPIs

- KPIs are calculated automatically for each search query and summary generation
- Steering vector KPIs (Style Adherence and Content Quality) are displayed for each summary
- See `KPI_DOCUMENTATION.md` for detailed KPI explanations and interpretation guides

## Troubleshooting

**Error: "Vector store not initialized"**
- Make sure you've run the pipeline at least once to build the vector store
- Or use `--rebuild` flag to rebuild it

**Error: "Data file not found"**
- Check that `Dell RAG compressed.txt` is in the current directory
- Or specify the correct path with `--data-file`

**Error: "Model loading failed"**
- Check your internet connection (model downloads on first use)
- TinyLlama doesn't require authentication, so this shouldn't be an auth issue
- Verify you have enough disk space (model is ~2.3GB)

**Error: "Out of memory"**
- Enable quantization: `--quantize` or set `USE_QUANTIZATION=true` in config
- Close other applications to free up RAM
- TinyLlama is already optimized for low memory usage

**Error: "Accelerate required"**
- Install accelerate: `pip install accelerate`
- Or disable quantization if you don't need it

**Model generation is slow**
- This is normal on CPU - expect 30-100 tokens/second
- Use quantization to speed up slightly
- Consider using GPU if available (set `--device cuda`)

**Steering vectors not working**
- Make sure the model is loaded successfully
- Check that steering-vectors library is installed: `pip install steering-vectors`
- First use will train vectors (takes a moment), subsequent uses are cached

## Configuration

Edit `config.py` to customize:

```python
# Model selection
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Device (auto, cpu, cuda)
MODEL_DEVICE = "auto"

# Quantization (for CPU efficiency)
USE_QUANTIZATION = True
QUANTIZATION_BITS = 8  # 8 or 4

# Steering vectors
STEERING_VECTOR_STRENGTH = 0.5  # 0.0 to 1.0
STEERING_VECTOR_LAYER = -1  # -1 for auto-select

# Hugging Face token (optional, not required for TinyLlama)
HF_TOKEN = None
```

## License

This project uses open-source models:
- TinyLlama: Apache 2.0 License
