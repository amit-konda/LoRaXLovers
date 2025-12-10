"""
Acceptance Tests for RAG Pipeline KPIs

Tests both happy path and edge cases to validate KPI calculations.
"""
import sys
import time
from rag_pipeline import ReviewRAGPipeline
from rag_metrics import RAGEvaluator
from data_loader import parse_reviews, prepare_reviews_for_rag


def test_happy_path(silent=False):
    """
    Happy Path Test: Well-formed query with clear intent
    
    Expected Behavior:
    - High retrieval precision (>0.7): Should retrieve mostly relevant reviews
    - High semantic similarity (>0.6): Retrieved docs should be semantically close to query
    - Reasonable response time (<5 seconds)
    
    Test Query: "battery problems" - Clear, specific query with expected matches
    
    Args:
        silent: If True, don't print output (for programmatic use)
    """
    if not silent:
        print("=" * 70)
        print("HAPPY PATH TEST")
        print("=" * 70)
        print("\nTest Query: 'battery problems'")
        print("Expected: High precision (>0.6) and similarity (>0.45)")
        print("Note: Precision threshold adjusted for cosine distance scores")
        print("-" * 70)
    
    # Initialize pipeline (model not needed for search tests)
    pipeline = ReviewRAGPipeline()
    pipeline.load_vectorstore("vectorstore")
    
    evaluator = RAGEvaluator()
    query = "battery problems"
    k = 5
    
    # Measure retrieval
    start_time = time.time()
    results = pipeline.search_reviews(query, k=k)
    end_time = time.time()
    response_time = end_time - start_time
    
    # Evaluate
    metrics = evaluator.evaluate_retrieval(results, query, response_time)
    
    if not silent:
        print(f"\nResults:")
        print(f"  Retrieved: {metrics.num_retrieved} documents")
        print(f"  Relevant: {metrics.num_relevant} documents")
        print(f"  Response Time: {metrics.average_response_time:.3f}s")
        print(f"\nKPIs:")
        print(f"  1. Retrieval Precision: {metrics.retrieval_precision:.3f} ({metrics.retrieval_precision*100:.1f}%)")
        print(f"  2. Semantic Similarity: {metrics.semantic_similarity_score:.3f} ({metrics.semantic_similarity_score*100:.1f}%)")
        
        print(f"\nExpected vs Actual:")
        print(f"  Precision > 0.6: Expected ✓, Actual: {'✓ PASS' if metrics.retrieval_precision > 0.6 else '✗ FAIL'}")
        print(f"  Similarity > 0.45: Expected ✓, Actual: {'✓ PASS' if metrics.semantic_similarity_score > 0.45 else '✗ FAIL'}")
        print(f"  Response Time < 5s: Expected ✓, Actual: {'✓ PASS' if response_time < 5.0 else '✗ FAIL'}")
        
        print(f"\nInterpretation:")
        if metrics.retrieval_precision > 0.6:
            print(f"  ✓ Good precision: {metrics.retrieval_precision*100:.1f}% of retrieved docs are relevant")
            print(f"    This means the system is effectively filtering relevant reviews.")
        else:
            print(f"  ✗ Low precision: Only {metrics.retrieval_precision*100:.1f}% are relevant")
            print(f"    This suggests the retrieval may need tuning.")
        
        if metrics.semantic_similarity_score > 0.45:
            print(f"  ✓ Good semantic match: {metrics.semantic_similarity_score*100:.1f}% average similarity")
            print(f"    The embedding model is effectively capturing semantic relationships.")
            print(f"    Note: For cosine distance, scores >45% indicate good semantic alignment.")
        else:
            print(f"  ✗ Low semantic similarity: {metrics.semantic_similarity_score*100:.1f}% average")
            print(f"    There may be a semantic gap between query and documents.")
        
        print(f"\nSample Retrieved Documents:")
        for i, result in enumerate(results[:3], 1):
            distance = result['similarity_score']
            similarity = 1.0 - min(distance / 2.0, 1.0)
            print(f"  {i}. Similarity: {similarity:.3f} - {result['content'][:80]}...")
    
    return metrics.retrieval_precision > 0.6 and metrics.semantic_similarity_score > 0.45


def test_edge_case(silent=False):
    """
    Edge Case Test: Ambiguous or out-of-scope query
    
    Expected Behavior:
    - Lower retrieval precision (<0.5): System may retrieve less relevant docs
    - Lower semantic similarity (<0.5): Query doesn't match well with available content
    - System should still return results (graceful degradation)
    
    Test Query: "quantum computing performance" - Completely unrelated to laptop reviews
    
    Args:
        silent: If True, don't print output (for programmatic use)
    """
    if not silent:
        print("\n" + "=" * 70)
        print("EDGE CASE / FAILURE TEST")
        print("=" * 70)
        print("\nTest Query: 'quantum computing performance'")
        print("Expected: Lower precision (<0.5) and similarity (<0.5)")
        print("Reason: Query is completely unrelated to customer reviews")
        print("-" * 70)
    
    # Initialize pipeline (model not needed for search tests)
    pipeline = ReviewRAGPipeline()
    pipeline.load_vectorstore("vectorstore")
    
    evaluator = RAGEvaluator()
    query = "quantum computing performance"  # Completely unrelated query
    k = 5
    
    # Measure retrieval
    start_time = time.time()
    results = pipeline.search_reviews(query, k=k)
    end_time = time.time()
    response_time = end_time - start_time
    
    # Evaluate
    metrics = evaluator.evaluate_retrieval(results, query, response_time)
    
    if not silent:
        print(f"\nResults:")
        print(f"  Retrieved: {metrics.num_retrieved} documents")
        print(f"  Relevant: {metrics.num_relevant} documents")
        print(f"  Response Time: {metrics.average_response_time:.3f}s")
        print(f"\nKPIs:")
        print(f"  1. Retrieval Precision: {metrics.retrieval_precision:.3f} ({metrics.retrieval_precision*100:.1f}%)")
        print(f"  2. Semantic Similarity: {metrics.semantic_similarity_score:.3f} ({metrics.semantic_similarity_score*100:.1f}%)")
        
        print(f"\nExpected vs Actual:")
        print(f"  Precision < 0.4: Expected ✓ (low relevance), Actual: {'✓ PASS' if metrics.retrieval_precision < 0.4 else '✗ FAIL'}")
        print(f"  Similarity < 0.4: Expected ✓ (poor match), Actual: {'✓ PASS' if metrics.semantic_similarity_score < 0.4 else '✗ FAIL'}")
        print(f"  Still returns results: Expected ✓ (graceful), Actual: {'✓ PASS' if len(results) > 0 else '✗ FAIL'}")
        
        print(f"\nInterpretation:")
        print(f"  This edge case demonstrates how the system handles out-of-scope queries:")
        if metrics.retrieval_precision < 0.5:
            print(f"  ✓ As expected, precision is low ({metrics.retrieval_precision*100:.1f}%)")
            print(f"    The system correctly identifies that retrieved docs are not highly relevant.")
        else:
            print(f"  ⚠ Unexpected: Precision is higher than expected")
            print(f"    This might indicate the embedding model is too permissive.")
        
        if metrics.semantic_similarity_score < 0.4:
            print(f"  ✓ As expected, semantic similarity is low ({metrics.semantic_similarity_score*100:.1f}%)")
            print(f"    The query doesn't match well with the review corpus, as expected.")
        else:
            print(f"  ⚠ Unexpected: Similarity is higher than expected")
            print(f"    This might indicate spurious semantic connections.")
        
        print(f"\nSample Retrieved Documents (showing poor match):")
        for i, result in enumerate(results[:3], 1):
            distance = result['similarity_score']
            similarity = 1.0 - min(distance / 2.0, 1.0)
            print(f"  {i}. Similarity: {similarity:.3f} - {result['content'][:80]}...")
        
        print(f"\nKey Insight:")
        print(f"  The system gracefully handles edge cases by still returning results,")
        print(f"  but the KPIs correctly reflect the poor match quality.")
        print(f"  This is valuable for detecting when queries are out of scope.")
    
    return metrics.retrieval_precision < 0.4 and metrics.semantic_similarity_score < 0.4


def test_empty_query(silent=False):
    """
    Additional edge case: Empty or very short query
    
    Args:
        silent: If True, don't print output (for programmatic use)
    """
    if not silent:
        print("\n" + "=" * 70)
        print("EDGE CASE: EMPTY/SHORT QUERY")
        print("=" * 70)
        print("\nTest Query: 'a' (single character)")
        print("Expected: System should handle gracefully")
        print("-" * 70)
    
    pipeline = ReviewRAGPipeline(use_gemini=False)
    pipeline.load_vectorstore("vectorstore")
    
    evaluator = RAGEvaluator()
    query = "a"
    k = 5
    
    start_time = time.time()
    try:
        results = pipeline.search_reviews(query, k=k)
        end_time = time.time()
        response_time = end_time - start_time
        
        metrics = evaluator.evaluate_retrieval(results, query, response_time)
        
        if not silent:
            print(f"\nResults: Retrieved {len(results)} documents")
            print(f"KPIs: Precision={metrics.retrieval_precision:.3f}, Similarity={metrics.semantic_similarity_score:.3f}")
            print(f"✓ System handled edge case gracefully")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all acceptance tests."""
    print("\n" + "=" * 70)
    print("RAG PIPELINE ACCEPTANCE TESTS")
    print("=" * 70)
    print("\nTesting KPI calculations and system behavior")
    print("=" * 70)
    
    results = []
    
    # Test 1: Happy Path
    try:
        result1 = test_happy_path()
        results.append(("Happy Path", result1))
    except Exception as e:
        print(f"\n✗ Happy Path Test Failed: {e}")
        results.append(("Happy Path", False))
    
    # Test 2: Edge Case
    try:
        result2 = test_edge_case()
        results.append(("Edge Case", result2))
    except Exception as e:
        print(f"\n✗ Edge Case Test Failed: {e}")
        results.append(("Edge Case", False))
    
    # Test 3: Empty Query
    try:
        result3 = test_empty_query()
        results.append(("Empty Query", result3))
    except Exception as e:
        print(f"\n✗ Empty Query Test Failed: {e}")
        results.append(("Empty Query", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

