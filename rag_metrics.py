"""
RAG Performance Metrics and KPIs
"""
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RAGMetrics:
    """Container for RAG performance metrics."""
    retrieval_precision: float
    semantic_similarity_score: float
    average_response_time: float
    retrieval_recall: Optional[float] = None
    num_retrieved: int = 0
    num_relevant: int = 0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'retrieval_precision': self.retrieval_precision,
            'semantic_similarity_score': self.semantic_similarity_score,
            'average_response_time': self.average_response_time,
            'retrieval_recall': self.retrieval_recall,
            'num_retrieved': self.num_retrieved,
            'num_relevant': self.num_relevant
        }


class RAGEvaluator:
    """
    Evaluator for RAG pipeline performance metrics.
    
    Tracks two key KPIs:
    1. Retrieval Precision - Measures how many retrieved documents are actually relevant
    2. Semantic Similarity Score - Measures how semantically similar retrieved docs are to the query
    """
    
    def __init__(self):
        self.query_times = []
        self.metrics_history = []
    
    def calculate_retrieval_precision(
        self, 
        retrieved_docs: List[Dict], 
        query: str,
        relevance_threshold: float = 0.45  # Adjusted for cosine distance scores
    ) -> Tuple[float, int, int]:
        """
        Calculate retrieval precision.
        
        KPI 1: Retrieval Precision
        Why it matters: Measures the quality of document retrieval. High precision means
        the system is retrieving mostly relevant documents, reducing noise in the context
        passed to the LLM. This directly impacts answer quality and reduces hallucination.
        
        Interpretation:
        - 1.0 (100%): All retrieved documents are highly relevant
        - 0.7-0.9: Good precision, most documents are relevant
        - 0.5-0.7: Moderate precision, some irrelevant documents retrieved
        - <0.5: Poor precision, many irrelevant documents
        
        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            query: Original query
            relevance_threshold: Similarity score threshold for relevance (0-1)
            
        Returns:
            Tuple of (precision, num_relevant, num_retrieved)
        """
        if not retrieved_docs:
            return 0.0, 0, 0
        
        num_retrieved = len(retrieved_docs)
        # Consider a document relevant if its similarity score is above threshold
        # Higher similarity = more relevant
        # We use (1 - score) because FAISS returns distance scores (lower = better)
        # So we convert to similarity: similarity = 1 - normalized_distance
        relevant_count = 0
        
        for doc in retrieved_docs:
            # Convert distance score to similarity score
            # FAISS similarity_search_with_score returns distance (lower is better)
            # We normalize: similarity = 1 / (1 + distance) or use 1 - normalized_distance
            distance = doc.get('similarity_score', 1.0)
            # Normalize distance to 0-1 range (assuming max distance ~2.0 for cosine)
            normalized_distance = min(distance / 2.0, 1.0)
            similarity = 1.0 - normalized_distance
            
            if similarity >= relevance_threshold:
                relevant_count += 1
        
        precision = relevant_count / num_retrieved if num_retrieved > 0 else 0.0
        return precision, relevant_count, num_retrieved
    
    def calculate_semantic_similarity_score(
        self,
        retrieved_docs: List[Dict],
        query: str
    ) -> float:
        """
        Calculate average semantic similarity score of retrieved documents.
        
        KPI 2: Semantic Similarity Score
        Why it matters: Measures how semantically close retrieved documents are to the query.
        Higher scores indicate better semantic understanding and retrieval quality. This KPI
        helps identify if the embedding model and vector search are working effectively.
        
        Interpretation:
        - 0.8-1.0: Excellent semantic match, documents are highly relevant
        - 0.6-0.8: Good semantic match, documents are relevant
        - 0.4-0.6: Moderate match, some semantic gap
        - <0.4: Poor semantic match, documents may not be relevant
        
        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            query: Original query
            
        Returns:
            Average semantic similarity score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        similarity_scores = []
        for doc in retrieved_docs:
            distance = doc.get('similarity_score', 1.0)
            # Convert distance to similarity
            # Normalize distance (assuming cosine distance range 0-2)
            normalized_distance = min(distance / 2.0, 1.0)
            similarity = 1.0 - normalized_distance
            similarity_scores.append(similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def measure_response_time(self, start_time: float, end_time: float) -> float:
        """Measure response time in seconds."""
        return end_time - start_time
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[Dict],
        query: str,
        response_time: float,
        relevance_threshold: float = 0.45
    ) -> RAGMetrics:
        """
        Evaluate retrieval performance and return metrics.
        
        Args:
            retrieved_docs: Retrieved documents
            query: Original query
            response_time: Time taken for retrieval in seconds
            relevance_threshold: Threshold for relevance
            
        Returns:
            RAGMetrics object with all KPIs
        """
        precision, num_relevant, num_retrieved = self.calculate_retrieval_precision(
            retrieved_docs, query, relevance_threshold
        )
        
        semantic_similarity = self.calculate_semantic_similarity_score(
            retrieved_docs, query
        )
        
        self.query_times.append(response_time)
        avg_response_time = np.mean(self.query_times) if self.query_times else response_time
        
        metrics = RAGMetrics(
            retrieval_precision=precision,
            semantic_similarity_score=semantic_similarity,
            average_response_time=avg_response_time,
            num_retrieved=num_retrieved,
            num_relevant=num_relevant
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all evaluations."""
        if not self.metrics_history:
            return {}
        
        precisions = [m.retrieval_precision for m in self.metrics_history]
        similarities = [m.semantic_similarity_score for m in self.metrics_history]
        times = [m.average_response_time for m in self.metrics_history]
        
        return {
            'avg_precision': np.mean(precisions),
            'avg_similarity': np.mean(similarities),
            'avg_time': np.mean(times),
            'total_queries': len(self.metrics_history),
            'min_precision': np.min(precisions),
            'max_precision': np.max(precisions),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }

