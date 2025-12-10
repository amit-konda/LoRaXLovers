"""
RAG Performance Metrics and KPIs
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Try to import sentence transformers for style adherence
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


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


@dataclass
class SteeringVectorMetrics:
    """Container for steering vector performance metrics."""
    style_adherence_score: float
    content_quality_score: float
    style: str
    steering_strength: float
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'style_adherence_score': self.style_adherence_score,
            'content_quality_score': self.content_quality_score,
            'style': self.style,
            'steering_strength': self.steering_strength
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
        query: str,  # pylint: disable=unused-argument
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
        query: str  # pylint: disable=unused-argument
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


class SteeringVectorEvaluator:
    """
    Evaluator for steering vector performance metrics.
    
    Tracks two key KPIs:
    1. Style Adherence Score - Measures how well output matches intended style
    2. Content Quality Preservation - Ensures steering doesn't degrade quality
    """
    
    def __init__(self):
        self.embedding_model = None
        self._init_embedding_model()
        self.metrics_history = []
    
    def _init_embedding_model(self):
        """Initialize embedding model for style adherence calculation."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight model for style comparison
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load embedding model for style metrics: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
    
    def _get_style_reference_texts(self, style: str) -> List[str]:
        """
        Get reference texts that exemplify each style.
        
        Args:
            style: Style name ("formal", "casual", "concise", "detailed")
            
        Returns:
            List of reference text examples
        """
        style_references = {
            "formal": [
                "The product demonstrates excellent performance characteristics.",
                "The device exhibits superior functionality and reliability.",
                "The customer experience was satisfactory and met expectations.",
                "This analysis provides comprehensive insights into the matter.",
                "The implementation demonstrates professional standards."
            ],
            "casual": [
                "This thing works great! Really happy with it.",
                "It's really good, no complaints here.",
                "It was okay I guess, nothing special.",
                "Love this product! Works exactly as expected.",
                "Pretty solid choice, would recommend."
            ],
            "concise": [
                "Good product. Fast delivery.",
                "Works well. Recommend.",
                "Excellent value. Satisfied.",
                "Meets expectations. Reliable.",
                "Quality item. Worth it."
            ],
            "detailed": [
                "This is an excellent product that was delivered very quickly and exceeded my expectations in multiple ways.",
                "The product functions as expected and I would highly recommend it to others based on my positive experience.",
                "After extensive use, I can confirm that this item provides outstanding value and performance across various use cases.",
                "The comprehensive features and reliable build quality make this a standout choice in its category.",
                "I've thoroughly tested this product and found it to deliver consistent results with excellent attention to detail."
            ],
            "balanced": [
                "The product performs well and meets expectations.",
                "Good quality item with reliable functionality.",
                "Satisfactory experience overall.",
                "Meets requirements and works as described.",
                "Solid product with good value."
            ]
        }
        return style_references.get(style, style_references["balanced"])
    
    def calculate_style_adherence_score(
        self,
        output_text: str,
        target_style: str
    ) -> float:
        """
        Calculate style adherence score.
        
        KPI 1: Style Adherence Score
        Why it matters: Measures how well the generated output matches the intended style
        (formal, casual, concise, detailed). High scores indicate effective steering.
        
        Interpretation:
        - 0.7-1.0: Excellent style match - Output clearly matches intended style
        - 0.5-0.7: Good style match - Output generally matches style
        - 0.3-0.5: Moderate match - Some style elements present
        - <0.3: Poor match - Output doesn't match intended style
        
        Args:
            output_text: Generated text to evaluate
            target_style: Intended style ("formal", "casual", "concise", "detailed", "balanced")
            
        Returns:
            Style adherence score (0-1)
        """
        if not output_text or not output_text.strip():
            return 0.0
        
        if target_style == "balanced":
            # Balanced style is neutral, so we measure deviation from extremes
            # Higher score means it's not too formal or too casual
            return 0.7  # Default good score for balanced
        
        # Get style reference texts
        style_refs = self._get_style_reference_texts(target_style)
        
        # Method 1: Embedding-based similarity (if available)
        if self.embedding_model is not None:
            try:
                # Get embeddings
                output_embedding = self.embedding_model.encode([output_text])[0]
                ref_embeddings = self.embedding_model.encode(style_refs)
                
                # Calculate cosine similarity to each reference
                similarities = []
                for ref_emb in ref_embeddings:
                    # Cosine similarity
                    dot_product = np.dot(output_embedding, ref_emb)
                    norm_output = np.linalg.norm(output_embedding)
                    norm_ref = np.linalg.norm(ref_emb)
                    if norm_output > 0 and norm_ref > 0:
                        similarity = dot_product / (norm_output * norm_ref)
                        similarities.append(similarity)
                
                if similarities:
                    # Average similarity to style references
                    avg_similarity = np.mean(similarities)
                    # Normalize to 0-1 range (cosine similarity is -1 to 1, but typically 0-1)
                    score = max(0.0, min(1.0, (avg_similarity + 1) / 2))
                    return float(score)
            except Exception as e:
                print(f"Warning: Embedding-based style calculation failed: {e}")
        
        # Method 2: Linguistic feature-based fallback
        return self._calculate_style_adherence_linguistic(output_text, target_style)
    
    def _calculate_style_adherence_linguistic(
        self,
        output_text: str,
        target_style: str
    ) -> float:
        """
        Calculate style adherence using linguistic features as fallback.
        
        Args:
            output_text: Generated text
            target_style: Target style
            
        Returns:
            Style adherence score (0-1)
        """
        # Extract linguistic features
        sentences = re.split(r'[.!?]+', output_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        total_words = len(output_text.split())
        
        # Style-specific features
        formal_indicators = ['demonstrates', 'exhibits', 'comprehensive', 'implementation', 
                            'characteristics', 'satisfactory', 'professional']
        casual_indicators = ['great', 'really', 'pretty', 'love', 'guess', 'thing', 'works']
        contraction_count = len(re.findall(r"\b\w+'\w+\b", output_text))
        exclamation_count = output_text.count('!')
        
        score = 0.5  # Base score
        
        if target_style == "formal":
            # Formal: longer sentences, formal vocabulary, no contractions
            formal_score = sum(1 for word in formal_indicators if word.lower() in output_text.lower()) / len(formal_indicators)
            length_score = min(1.0, avg_sentence_length / 20.0)  # Longer sentences = more formal
            no_contractions = 1.0 if contraction_count == 0 else max(0.0, 1.0 - contraction_count * 0.2)
            score = (formal_score * 0.4 + length_score * 0.3 + no_contractions * 0.3)
            
        elif target_style == "casual":
            # Casual: shorter sentences, casual vocabulary, contractions, exclamations
            casual_score = sum(1 for word in casual_indicators if word.lower() in output_text.lower()) / len(casual_indicators)
            length_score = max(0.0, 1.0 - avg_sentence_length / 15.0)  # Shorter = more casual
            has_contractions = min(1.0, contraction_count * 0.3)
            has_exclamations = min(1.0, exclamation_count * 0.2)
            score = (casual_score * 0.3 + length_score * 0.3 + has_contractions * 0.2 + has_exclamations * 0.2)
            
        elif target_style == "concise":
            # Concise: very short sentences, minimal words
            length_score = max(0.0, 1.0 - avg_sentence_length / 10.0)  # Very short = concise
            word_count_score = max(0.0, 1.0 - total_words / 50.0)  # Fewer words = concise
            score = (length_score * 0.6 + word_count_score * 0.4)
            
        elif target_style == "detailed":
            # Detailed: longer sentences, more words, comprehensive
            length_score = min(1.0, avg_sentence_length / 25.0)  # Longer = detailed
            word_count_score = min(1.0, total_words / 100.0)  # More words = detailed
            comprehensive_words = sum(1 for word in ['comprehensive', 'extensive', 'thorough', 'detailed', 'multiple'] 
                                     if word.lower() in output_text.lower())
            comprehensive_score = min(1.0, comprehensive_words * 0.3)
            score = (length_score * 0.4 + word_count_score * 0.4 + comprehensive_score * 0.2)
        
        return float(max(0.0, min(1.0, score)))
    
    def calculate_content_quality_score(
        self,
        output_text: str,
        source_reviews: List[Dict],
        query: str
    ) -> float:
        """
        Calculate content quality preservation score.
        
        KPI 2: Content Quality Preservation
        Why it matters: Ensures that steering vectors don't degrade the factual accuracy,
        coherence, or relevance of the generated content. High scores indicate that steering
        maintains content quality while applying style.
        
        Interpretation:
        - 0.7-1.0: Excellent quality - Content is coherent, relevant, and well-structured
        - 0.5-0.7: Good quality - Content is generally coherent and relevant
        - 0.3-0.5: Moderate quality - Some coherence or relevance issues
        - <0.3: Poor quality - Significant quality degradation
        
        Args:
            output_text: Generated summary text
            source_reviews: Source reviews used for summarization
            query: Original query
            
        Returns:
            Content quality score (0-1)
        """
        if not output_text or not output_text.strip():
            return 0.0
        
        scores = []
        
        # 1. Coherence score (sentence structure and flow)
        coherence_score = self._calculate_coherence(output_text)
        scores.append(coherence_score)
        
        # 2. Relevance score (semantic similarity to source content)
        relevance_score = self._calculate_relevance(output_text, source_reviews, query)
        scores.append(relevance_score)
        
        # 3. Information density (not too sparse, not too verbose)
        density_score = self._calculate_information_density(output_text, source_reviews)
        scores.append(density_score)
        
        # 4. Structure score (proper formatting, complete sentences)
        structure_score = self._calculate_structure_quality(output_text)
        scores.append(structure_score)
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Coherence and relevance are most important
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        return float(max(0.0, min(1.0, final_score)))
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5  # Single sentence, can't assess coherence
        
        # Check for transition words (indicates coherence)
        transition_words = ['however', 'therefore', 'furthermore', 'additionally', 
                          'moreover', 'consequently', 'thus', 'also', 'and', 'but']
        transition_count = sum(1 for word in transition_words 
                              if word.lower() in text.lower())
        transition_score = min(1.0, transition_count / 3.0)
        
        # Check sentence length consistency (very inconsistent = less coherent)
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            length_std = np.std(sentence_lengths)
            avg_length = np.mean(sentence_lengths)
            consistency_score = max(0.0, 1.0 - (length_std / max(avg_length, 1.0)) * 0.5)
        else:
            consistency_score = 0.7
        
        # Check for repetition (high repetition = less coherent)
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            repetition_score = unique_ratio
        else:
            repetition_score = 0.5
        
        return (transition_score * 0.3 + consistency_score * 0.4 + repetition_score * 0.3)
    
    def _calculate_relevance(self, output_text: str, source_reviews: List[Dict], query: str) -> float:
        """Calculate relevance to source content and query."""
        if not source_reviews:
            return 0.5
        
        # Extract key terms from query
        query_terms = set(query.lower().split())
        
        # Extract key terms from source reviews
        source_text = " ".join([r.get('content', '') for r in source_reviews])
        source_terms = set(source_text.lower().split())
        
        # Extract key terms from output
        output_terms = set(output_text.lower().split())
        
        # Calculate overlap with query
        query_overlap = len(query_terms & output_terms) / max(len(query_terms), 1)
        
        # Calculate overlap with source (sample of common terms)
        common_source_terms = source_terms & output_terms
        source_overlap = len(common_source_terms) / max(len(output_terms), 1)
        
        # Use embedding similarity if available
        if self.embedding_model is not None:
            try:
                # Compare output to source content
                source_combined = " ".join([r.get('content', '')[:200] for r in source_reviews[:3]])
                if source_combined:
                    embeddings = self.embedding_model.encode([output_text, source_combined])
                    dot_product = np.dot(embeddings[0], embeddings[1])
                    norm_0 = np.linalg.norm(embeddings[0])
                    norm_1 = np.linalg.norm(embeddings[1])
                    if norm_0 > 0 and norm_1 > 0:
                        semantic_similarity = dot_product / (norm_0 * norm_1)
                        semantic_score = max(0.0, (semantic_similarity + 1) / 2)
                    else:
                        semantic_score = 0.5
                else:
                    semantic_score = 0.5
            except Exception:
                semantic_score = 0.5
        else:
            semantic_score = 0.5
        
        # Weighted combination
        return (query_overlap * 0.3 + source_overlap * 0.3 + semantic_score * 0.4)
    
    def _calculate_information_density(self, output_text: str, source_reviews: List[Dict]) -> float:
        """Calculate information density (not too sparse, not too verbose)."""
        if not source_reviews:
            return 0.5
        
        output_words = len(output_text.split())
        source_words = sum(len(r.get('content', '').split()) for r in source_reviews)
        
        if source_words == 0:
            return 0.5
        
        # Ideal compression ratio: 10-30% of source (summary should be concise but informative)
        compression_ratio = output_words / max(source_words, 1)
        
        if 0.1 <= compression_ratio <= 0.3:
            return 1.0  # Ideal range
        elif 0.05 <= compression_ratio < 0.1 or 0.3 < compression_ratio <= 0.5:
            return 0.7  # Acceptable
        elif compression_ratio < 0.05:
            return max(0.3, compression_ratio * 10)  # Too sparse
        else:
            return max(0.3, 1.0 - (compression_ratio - 0.5) * 0.5)  # Too verbose
    
    def _calculate_structure_quality(self, text: str) -> float:
        """Calculate structural quality (formatting, sentence completeness)."""
        if not text:
            return 0.0
        
        scores = []
        
        # Check for proper sentence endings
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s for s in sentences if s.strip() and len(s.strip().split()) >= 3]
        sentence_completeness = len(complete_sentences) / max(len(sentences), 1)
        scores.append(sentence_completeness)
        
        # Check for proper capitalization (first letter of sentences)
        first_chars = [s.strip()[0] if s.strip() else '' for s in sentences if s.strip()]
        if first_chars:
            capitalized = sum(1 for c in first_chars if c.isupper())
            capitalization_score = capitalized / len(first_chars)
        else:
            capitalization_score = 0.5
        scores.append(capitalization_score)
        
        # Check for reasonable paragraph structure (not one giant block)
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1 or len(text) < 500:
            paragraph_score = 1.0
        else:
            # Single long paragraph - penalize slightly
            paragraph_score = 0.7
        scores.append(paragraph_score)
        
        return np.mean(scores)
    
    def evaluate_steering_vector(
        self,
        output_text: str,
        target_style: str,
        source_reviews: List[Dict],
        query: str,
        steering_strength: float
    ) -> SteeringVectorMetrics:
        """
        Evaluate steering vector performance and return metrics.
        
        Args:
            output_text: Generated summary text
            target_style: Intended style
            source_reviews: Source reviews used
            query: Original query
            steering_strength: Steering strength used (0.0-1.0)
            
        Returns:
            SteeringVectorMetrics object with all KPIs
        """
        style_adherence = self.calculate_style_adherence_score(output_text, target_style)
        content_quality = self.calculate_content_quality_score(output_text, source_reviews, query)
        
        metrics = SteeringVectorMetrics(
            style_adherence_score=style_adherence,
            content_quality_score=content_quality,
            style=target_style,
            steering_strength=steering_strength
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all steering vector evaluations."""
        if not self.metrics_history:
            return {}
        
        style_scores = [m.style_adherence_score for m in self.metrics_history]
        quality_scores = [m.content_quality_score for m in self.metrics_history]
        
        return {
            'avg_style_adherence': np.mean(style_scores),
            'avg_content_quality': np.mean(quality_scores),
            'total_evaluations': len(self.metrics_history),
            'min_style_adherence': np.min(style_scores),
            'max_style_adherence': np.max(style_scores),
            'min_content_quality': np.min(quality_scores),
            'max_content_quality': np.max(quality_scores)
        }

