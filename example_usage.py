"""
Example usage of the RAG pipeline for customer reviews.
"""
from data_loader import parse_reviews, prepare_reviews_for_rag
from rag_pipeline import ReviewRAGPipeline


def example_search():
    """Example: Search for reviews about battery issues."""
    print("=" * 60)
    print("Example 1: Searching for reviews about battery issues")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ReviewRAGPipeline()
    
    # Load vector store (or build if it doesn't exist)
    try:
        pipeline.load_vectorstore("vectorstore")
    except:
        print("Building vector store...")
        reviews = prepare_reviews_for_rag(parse_reviews("Dell RAG compressed.txt"))
        pipeline.build_vectorstore(reviews, save_path="vectorstore")
    
    # Search
    results = pipeline.search_reviews("battery life problems", k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {result['similarity_score']:.4f}")
        print(f"Content: {result['content'][:200]}...")


def example_summarize():
    """Example: Summarize reviews about overheating."""
    print("\n" + "=" * 60)
    print("Example 2: Summarizing reviews about overheating")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ReviewRAGPipeline()
    
    # Load vector store
    try:
        pipeline.load_vectorstore("vectorstore")
    except:
        print("Building vector store...")
        reviews = prepare_reviews_for_rag(parse_reviews("Dell RAG compressed.txt"))
        pipeline.build_vectorstore(reviews, save_path="vectorstore")
    
    # Summarize
    summary = pipeline.summarize_reviews("overheating problems", k=5)
    print(summary)


if __name__ == "__main__":
    example_search()
    example_summarize()

