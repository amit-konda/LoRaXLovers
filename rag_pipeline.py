"""
RAG Pipeline for searching and summarizing customer reviews.
"""
import os
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Use google-generativeai directly (more reliable)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class ReviewRAGPipeline:
    """
    RAG pipeline for customer review search and summarization.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_gemini: bool = True,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the sentence transformer model
            use_gemini: Whether to use Google Gemini for LLM (default: True)
            gemini_api_key: Google Gemini API key
            gemini_model: Gemini model name (default: gemini-pro)
        """
        self.embedding_model = embedding_model
        self.use_gemini = use_gemini
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-pro")
        
        # Initialize embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.vectorstore = None
        self.qa_chain = None
        
        # Set Gemini API key if provided
        if self.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
    
    def build_vectorstore(self, reviews: List[Dict], save_path: Optional[str] = None):
        """
        Build the vector store from reviews.
        
        Args:
            reviews: List of review dictionaries
            save_path: Optional path to save the vector store
        """
        print(f"Processing {len(reviews)} reviews...")
        
        # Create documents
        documents = []
        for review in reviews:
            doc = Document(
                page_content=review['text'],
                metadata=review.get('metadata', {})
            )
            documents.append(doc)
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        texts = self.text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks")
        
        # Create vector store
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        print("Vector store created successfully!")
        
        # Save if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.vectorstore.save_local(save_path)
            print(f"Vector store saved to {save_path}")
    
    def load_vectorstore(self, load_path: str):
        """
        Load a previously saved vector store.
        
        Args:
            load_path: Path to the saved vector store
        """
        print(f"Loading vector store from {load_path}...")
        try:
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            # Try without allow_dangerous_deserialization for newer versions
            self.vectorstore = FAISS.load_local(load_path, self.embeddings)
        print("Vector store loaded successfully!")
    
    def search_reviews(self, query: str, k: int = 5, return_metrics: bool = False) -> List[Dict]:
        """
        Search for relevant reviews.
        
        Args:
            query: Search query
            k: Number of results to return
            return_metrics: If True, also return performance metrics
            
        Returns:
            List of relevant review dictionaries, optionally with metrics
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() first.")
        
        # Search for similar documents
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        if return_metrics:
            from rag_metrics import RAGEvaluator
            import time
            evaluator = RAGEvaluator()
            # Note: response_time would need to be measured outside this function
            # For now, we'll just return results with metrics capability
            return results, evaluator
        
        return results
    
    def summarize_reviews(self, query: str, k: int = 5) -> str:
        """
        Summarize reviews related to a query.
        
        Args:
            query: Search query
            k: Number of reviews to consider for summarization
            
        Returns:
            Summary string
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() first.")
        
        # Get relevant reviews
        relevant_reviews = self.search_reviews(query, k=k)
        
        if not relevant_reviews:
            return "No relevant reviews found."
        
        # Combine review texts
        review_texts = "\n\n---\n\n".join([
            f"Review {i+1} (Rating: {r['metadata'].get('rating', 'N/A')} stars):\n{r['content']}"
            for i, r in enumerate(relevant_reviews)
        ])
        
        # Create summary prompt
        summary_prompt = f"""Based on the following customer reviews, provide a comprehensive summary that addresses the query: "{query}"

Reviews:
{review_texts}

Please provide a summary that:
1. Addresses the main points related to the query
2. Highlights common themes and patterns
3. Mentions the overall sentiment (positive/negative/neutral)
4. Includes specific examples when relevant

Summary:"""
        
        # Generate summary using Google Gemini
        if self.use_gemini and self.gemini_api_key:
            try:
                if not GEMINI_AVAILABLE:
                    raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
                
                # Use REST API directly (most reliable)
                import requests
                
                # Use the correct model name format for Gemini API
                # Model names should be like: gemini-2.5-flash, gemini-2.5-pro, etc.
                model_name = self.gemini_model
                # Ensure model name doesn't have "models/" prefix
                if model_name.startswith("models/"):
                    model_name = model_name.replace("models/", "")
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{"text": summary_prompt}]
                    }]
                }
                
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result['candidates'][0]['content']['parts'][0]['text']
                else:
                    error_text = response.text[:500]
                    raise Exception(f"API returned {response.status_code}: {error_text}")
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error using Gemini API: {error_msg}")
                if "401" in error_msg or "invalid" in error_msg.lower() or "API_KEY" in error_msg:
                    print("💡 Gemini API key authentication failed. Verify your API key in config.py")
                elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                    print("💡 API quota exceeded. Check your Gemini API usage limits.")
                else:
                    print(f"💡 Error details: {error_msg}")
                # Fallback to simple summarization
                print("📝 Falling back to simple summarization...")
                summary = self._simple_summary(relevant_reviews, query)
        else:
            # Use simple summarization
            summary = self._simple_summary(relevant_reviews, query)
        
        return summary
    
    def _simple_summary(self, reviews: List[Dict], query: str) -> str:
        """
        Create a simple summary without LLM.
        
        Args:
            reviews: List of review dictionaries
            query: Original query
            
        Returns:
            Summary string
        """
        if not reviews:
            return "No reviews found."
        
        # Extract ratings
        ratings = [r['metadata'].get('rating', 'N/A') for r in reviews]
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        # Count reviews
        total_reviews = len(reviews)
        
        # Extract key phrases from reviews
        summary_parts = [
            f"Found {total_reviews} relevant reviews for: '{query}'",
            f"\nRating distribution: {dict(rating_counts)}",
            f"\n\nKey Reviews:"
        ]
        
        for i, review in enumerate(reviews[:3], 1):
            rating = review['metadata'].get('rating', 'N/A')
            content = review['content'][:300] + "..." if len(review['content']) > 300 else review['content']
            summary_parts.append(f"\n\nReview {i} ({rating} stars):\n{content}")
        
        return "\n".join(summary_parts)
