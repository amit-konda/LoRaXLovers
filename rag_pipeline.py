"""
RAG Pipeline for searching and summarizing customer reviews.
"""
import os
import torch
from typing import List, Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Try to import transformers and steering vectors
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from steering_vectors import train_steering_vector
    STEERING_VECTORS_AVAILABLE = True
except ImportError:
    STEERING_VECTORS_AVAILABLE = False
    train_steering_vector = None


class ReviewRAGPipeline:
    """
    RAG pipeline for customer review search and summarization.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_quantization: Optional[bool] = None,
        quantization_bits: int = 8,
        steering_strength: float = 0.5,
        steering_layer: int = -1,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the sentence transformer model
            model_name: Hugging Face model identifier (default from config)
            device: Device to use ("cuda", "cpu", or "auto")
            use_quantization: Whether to use quantization (default from config)
            quantization_bits: Bits for quantization (8 or 4)
            steering_strength: Strength of steering vectors (0.0 to 1.0)
            steering_layer: Layer to apply steering (-1 for auto)
            hf_token: Hugging Face token for model access
        """
        self.embedding_model = embedding_model
        
        # Load config defaults
        try:
            from config import (
                MODEL_NAME, MODEL_DEVICE, USE_QUANTIZATION,
                QUANTIZATION_BITS, STEERING_VECTOR_STRENGTH,
                STEERING_VECTOR_LAYER, HF_TOKEN
            )
            self.model_name = model_name or MODEL_NAME
            self.device = device or MODEL_DEVICE
            self.use_quantization = use_quantization if use_quantization is not None else USE_QUANTIZATION
            self.quantization_bits = quantization_bits or QUANTIZATION_BITS
            self.steering_strength = steering_strength or STEERING_VECTOR_STRENGTH
            self.steering_layer = steering_layer if steering_layer != -1 else STEERING_VECTOR_LAYER
            self.hf_token = hf_token or HF_TOKEN
        except ImportError:
            # Fallback defaults
            self.model_name = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.device = device or "auto"
            self.use_quantization = use_quantization if use_quantization is not None else True
            self.quantization_bits = quantization_bits or 8
            self.steering_strength = steering_strength or 0.5
            self.steering_layer = steering_layer if steering_layer != -1 else -1
            self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        self.model = None
        self.tokenizer = None
        self.steering_vectors = {}  # Cache for steering vector objects
        
        # Load model (lazy loading - will load on first use)
        self._model_loaded = False
    
    def _load_model_with_quantization(self):
        """Load the language model with optional quantization."""
        if self._model_loaded:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers library not available. Model will not be loaded.")
            return
        
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        
        try:
            # Configure quantization if enabled
            quantization_config = None
            if self.use_quantization and self.device == "cpu":
                if self.quantization_bits == 8:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                elif self.quantization_bits == 4:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
            
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set padding side to left for generation
            self.tokenizer.padding_side = "left"
            
            # Load model
            print("Loading model (this may take a while on first run)...")
            model_kwargs = {
                "token": self.hf_token,
                "trust_remote_code": True,
            }
            
            # Check if accelerate is available for quantization
            try:
                import accelerate
                accelerate_available = True
            except ImportError:
                accelerate_available = False
                if quantization_config:
                    print("⚠️ Accelerate not available, disabling quantization")
                    quantization_config = None
            
            if quantization_config and accelerate_available:
                # Quantization requires device_map="auto" and accelerate
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                # Load without quantization
                model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                # For CPU, don't use device_map or low_cpu_mem_usage (requires accelerate)
                # Just load normally and move to device manually
                if self.device != "cpu":
                    model_kwargs["device_map"] = self.device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not (quantization_config and accelerate_available):
                if hasattr(self.model, 'device'):
                    # Model already has device attribute
                    pass
                else:
                    # Manually move to device
                    self.model = self.model.to(self.device)
            
            # Set to eval mode for inference
            self.model.eval()
            
            self._model_loaded = True
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Falling back to simple summarization")
            self.model = None
            self.tokenizer = None
    
    def _get_style_steering_vector(self, style: str = "balanced"):
        """
        Get or create a steering vector object for the specified style.
        
        Args:
            style: Style preset ("formal", "casual", "concise", "detailed", "balanced")
            
        Returns:
            Steering vector object (with .apply() method) or None
        """
        if not STEERING_VECTORS_AVAILABLE or self.model is None:
            return None
        
        # Return cached vector if available
        if style in self.steering_vectors:
            return self.steering_vectors[style]
        
        # For balanced, return None (no steering)
        if style == "balanced":
            return None
        
        try:
            # Define contrastive pairs for style control
            style_pairs = {
                "formal": [
                    ("The product demonstrates excellent performance", "This thing works great!"),
                    ("The device exhibits superior functionality", "It's really good"),
                    ("The customer experience was satisfactory", "It was okay I guess"),
                ],
                "casual": [
                    ("This thing works great!", "The product demonstrates excellent performance"),
                    ("It's really good", "The device exhibits superior functionality"),
                    ("It was okay I guess", "The customer experience was satisfactory"),
                ],
                "concise": [
                    ("Good product. Fast delivery.", "This is an excellent product that was delivered very quickly and exceeded my expectations."),
                    ("Works well. Recommend.", "The product functions as expected and I would highly recommend it to others."),
                ],
                "detailed": [
                    ("This is an excellent product that was delivered very quickly and exceeded my expectations.", "Good product. Fast delivery."),
                    ("The product functions as expected and I would highly recommend it to others.", "Works well. Recommend."),
                ],
            }
            
            if style not in style_pairs:
                return None
            
            print(f"Training steering vector for style: {style}")
            pairs = style_pairs[style]
            
            # Train steering vector
            steering_vector = train_steering_vector(
                self.model,
                self.tokenizer,
                pairs,
                show_progress=True,
            )
            
            # Cache the vector
            self.steering_vectors[style] = steering_vector
            return steering_vector
            
        except Exception as e:
            print(f"⚠️ Error creating steering vector for {style}: {e}")
            return None
    
    def _apply_steering_vector(self, steering_vector_obj, multiplier: float = None):
        """
        Apply steering vector to model using the library's context manager.
        
        Args:
            steering_vector_obj: The steering vector object returned by train_steering_vector
            multiplier: Optional multiplier to adjust steering strength (defaults to self.steering_strength)
        """
        if steering_vector_obj is None or self.model is None:
            return None
        
        # Use the steering vector's apply method as a context manager
        # The multiplier controls the strength of steering
        multiplier = multiplier if multiplier is not None else self.steering_strength
        return steering_vector_obj.apply(self.model, multiplier=multiplier)
    
    def _generate_with_steering(
        self,
        prompt: str,
        style: str = "balanced",
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text with steering vector applied using the library's context manager.
        
        Args:
            prompt: Input prompt
            style: Style preset
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Cannot generate text.")
        
        # Get steering vector object
        steering_vector_obj = self._get_style_steering_vector(style)
        
        # Apply steering using the library's context manager
        # This automatically handles hook registration and cleanup
        steering_context = self._apply_steering_vector(steering_vector_obj, multiplier=self.steering_strength)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        # Get device from model
        model_device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device(self.device)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate within the steering context
        if steering_context is not None:
            # Use context manager if steering is applied
            with steering_context:
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
        else:
            # No steering, generate normally
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        # Handle both regular prompts and chat template prompts
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        else:
            # For chat templates, try to extract just the assistant response
            # Look for common chat markers
            if "<|assistant|>" in generated_text:
                parts = generated_text.split("<|assistant|>")
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
            elif "### Assistant:" in generated_text or "Assistant:" in generated_text:
                # Try to find the assistant response
                for marker in ["### Assistant:", "Assistant:"]:
                    if marker in generated_text:
                        parts = generated_text.split(marker)
                        if len(parts) > 1:
                            generated_text = parts[-1].strip()
                            break
        
        return generated_text
    
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
    
    def summarize_reviews(
        self,
        query: str,
        k: int = 5,
        style: str = "balanced",
        max_tokens: int = 500
    ) -> str:
        """
        Summarize reviews related to a query.
        
        Args:
            query: Search query
            k: Number of reviews to consider for summarization
            style: Style preset ("formal", "casual", "concise", "detailed", "balanced")
            max_tokens: Maximum tokens for summary
            
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
        style_instructions = {
            "formal": "Use formal, professional language.",
            "casual": "Use casual, conversational language.",
            "concise": "Be brief and to the point.",
            "detailed": "Provide comprehensive details and explanations.",
            "balanced": ""
        }
        style_instruction = style_instructions.get(style, "")
        
        # Format prompt for TinyLlama chat format if needed
        base_prompt = f"""Based on the following customer reviews, provide a comprehensive summary that addresses the query: "{query}"

Reviews:
{review_texts}

Please provide a summary that:
1. Addresses the main points related to the query
2. Highlights common themes and patterns
3. Mentions the overall sentiment (positive/negative/neutral)
4. Includes specific examples when relevant
{style_instruction}

Summary:"""
        
        # Use chat template if available (for TinyLlama and other chat models)
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Try to use chat template
                messages = [{"role": "user", "content": base_prompt}]
                summary_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback to regular prompt if chat template fails
                summary_prompt = base_prompt
        else:
            summary_prompt = base_prompt
        
        # Try to use model for summarization
        try:
            # Load model if not already loaded
            if not self._model_loaded:
                self._load_model_with_quantization()
            
            if self.model is not None and self.tokenizer is not None:
                print(f"Generating summary with {self.model_name} (style: {style})...")
                summary = self._generate_with_steering(
                    summary_prompt,
                    style=style,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return summary
            else:
                raise ValueError("Model not available")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error using model: {error_msg}")
            print("📝 Falling back to simple summarization...")
            # Fallback to simple summarization
            return self._simple_summary(relevant_reviews, query)
    
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
