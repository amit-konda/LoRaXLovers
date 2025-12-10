"""
Main application for RAG-based customer review search and summarization.
"""
import argparse
import os
import sys
from data_loader import parse_reviews, prepare_reviews_for_rag
from rag_pipeline import ReviewRAGPipeline

# Try to load config if available
try:
    from config import MODEL_NAME, MODEL_DEVICE, USE_QUANTIZATION, HF_TOKEN
    if HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", HF_TOKEN)
        os.environ.setdefault("HUGGINGFACE_TOKEN", HF_TOKEN)
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for Customer Review Search and Summarization"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="Dell RAG compressed.txt",
        help="Path to the reviews data file"
    )
    parser.add_argument(
        "--vectorstore-path",
        type=str,
        default="vectorstore",
        help="Path to save/load the vector store"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store even if it exists"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["search", "summarize", "interactive"],
        default="interactive",
        help="Mode: search, summarize, or interactive"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default from config.py, e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=None,
        help="Enable 8-bit quantization (recommended for CPU)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_false",
        dest="quantize",
        help="Disable quantization"
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        help="Steering vector strength (0.0 to 1.0, default: 0.5)"
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["balanced", "formal", "casual", "concise", "detailed"],
        default="balanced",
        help="Summary style preset (default: balanced)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for model access (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    pipeline = ReviewRAGPipeline(
        model_name=args.model,
        device=args.device,
        use_quantization=args.quantize if hasattr(args, 'quantize') else None,
        steering_strength=args.steering_strength,
        hf_token=hf_token
    )
    
    # Build or load vector store
    vectorstore_exists = os.path.exists(args.vectorstore_path) and os.path.isdir(args.vectorstore_path)
    
    if args.rebuild or not vectorstore_exists:
        print("Building vector store from reviews...")
        if not os.path.exists(args.data_file):
            print(f"Error: Data file '{args.data_file}' not found!")
            sys.exit(1)
        
        # Load and prepare reviews
        print(f"Loading reviews from {args.data_file}...")
        raw_reviews = parse_reviews(args.data_file)
        print(f"Loaded {len(raw_reviews)} reviews")
        
        reviews = prepare_reviews_for_rag(raw_reviews)
        print(f"Prepared {len(reviews)} reviews for RAG")
        
        # Build vector store
        pipeline.build_vectorstore(reviews, save_path=args.vectorstore_path)
    else:
        print(f"Loading existing vector store from {args.vectorstore_path}...")
        pipeline.load_vectorstore(args.vectorstore_path)
    
    # Execute query
    if args.mode == "interactive":
        interactive_mode(pipeline, args.k)
    elif args.mode == "search":
        if not args.query:
            print("Error: --query is required for search mode")
            sys.exit(1)
        search_reviews(pipeline, args.query, args.k)
    elif args.mode == "summarize":
        if not args.query:
            print("Error: --query is required for summarize mode")
            sys.exit(1)
        summarize_reviews(pipeline, args.query, args.k, args.style)


def search_reviews(pipeline: ReviewRAGPipeline, query: str, k: int):
    """Search for relevant reviews."""
    print(f"\n{'='*60}")
    print(f"Searching for: '{query}'")
    print(f"{'='*60}\n")
    
    results = pipeline.search_reviews(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n{'─'*60}")
        print(f"Result {i} (Similarity Score: {result['similarity_score']:.4f})")
        print(f"{'─'*60}")
        print(result['content'])
        print(f"\nMetadata: {result['metadata']}")


def summarize_reviews(pipeline: ReviewRAGPipeline, query: str, k: int, style: str = "balanced"):
    """Summarize relevant reviews."""
    print(f"\n{'='*60}")
    print(f"Summarizing reviews for: '{query}'")
    print(f"Style: {style}")
    print(f"{'='*60}\n")
    
    summary = pipeline.summarize_reviews(query, k=k, style=style)
    print(summary)


def interactive_mode(pipeline: ReviewRAGPipeline, k: int):
    """Interactive mode for querying reviews."""
    print("\n" + "="*60)
    print("Customer Review RAG Pipeline - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  - Type a search query to find relevant reviews")
    print("  - Type 'search: <query>' to search for reviews")
    print("  - Type 'summarize: <query>' to get a summary")
    print("  - Type 'summarize: <query> --style <style>' to get a styled summary")
    print("    (styles: balanced, formal, casual, concise, detailed)")
    print("  - Type 'quit' or 'exit' to exit")
    print("\n" + "="*60 + "\n")
    
    while True:
        try:
            user_input = input("\nEnter your query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.startswith('search:'):
                query = user_input[7:].strip()
                search_reviews(pipeline, query, k)
            elif user_input.startswith('summarize:'):
                parts = user_input[10:].strip().split('--style')
                query = parts[0].strip()
                style = parts[1].strip() if len(parts) > 1 else "balanced"
                summarize_reviews(pipeline, query, k, style)
            else:
                # Default to search
                search_reviews(pipeline, user_input, k)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
