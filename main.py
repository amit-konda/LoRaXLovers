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
    from config import GEMINI_API_KEY, GEMINI_MODEL
    if GEMINI_API_KEY:
        os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)
    if GEMINI_MODEL:
        os.environ.setdefault("GEMINI_MODEL", GEMINI_MODEL)
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
        "--use-gemini",
        action="store_true",
        default=True,
        help="Use Google Gemini for summarization (default: enabled)"
    )
    parser.add_argument(
        "--no-gemini",
        action="store_false",
        dest="use_gemini",
        help="Disable Gemini and use simple summarization"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY env var, default from config.py)"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        help="Gemini model name (default: gemini-pro)"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    gemini_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    gemini_model = args.gemini_model or os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # Default to using Gemini if API key is available
    use_gemini = args.use_gemini and gemini_key is not None
    
    pipeline = ReviewRAGPipeline(
        use_gemini=use_gemini,
        gemini_api_key=gemini_key,
        gemini_model=gemini_model
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
        summarize_reviews(pipeline, args.query, args.k)


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


def summarize_reviews(pipeline: ReviewRAGPipeline, query: str, k: int):
    """Summarize relevant reviews."""
    print(f"\n{'='*60}")
    print(f"Summarizing reviews for: '{query}'")
    print(f"{'='*60}\n")
    
    summary = pipeline.summarize_reviews(query, k=k)
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
                query = user_input[10:].strip()
                summarize_reviews(pipeline, query, k)
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
