"""
Streamlit Dashboard for Customer Review RAG Pipeline
"""
import streamlit as st
import os
import time
import sys
from io import StringIO
from contextlib import redirect_stdout
from data_loader import parse_reviews, prepare_reviews_for_rag
from rag_pipeline import ReviewRAGPipeline
from rag_metrics import RAGEvaluator
from acceptance_tests import test_happy_path, test_edge_case, test_empty_query

# Try to load config
try:
    from config import GEMINI_API_KEY, GEMINI_MODEL
    if GEMINI_API_KEY:
        os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)
    if GEMINI_MODEL:
        os.environ.setdefault("GEMINI_MODEL", GEMINI_MODEL)
except ImportError:
    pass

# Page configuration
st.set_page_config(
    page_title="Customer Review RAG Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(use_gemini=True):
    """Load the RAG pipeline (cached for performance)."""
    # Get Gemini API key from config or environment
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    pipeline = ReviewRAGPipeline(
        use_gemini=use_gemini and gemini_key is not None,
        gemini_api_key=gemini_key,
        gemini_model=gemini_model
    )
    
    vectorstore_path = "vectorstore"
    
    # Check if vector store exists
    if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
        try:
            pipeline.load_vectorstore(vectorstore_path)
            return pipeline, True
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return None, False
    else:
        # Build vector store
        data_file = "Dell RAG compressed.txt"
        if not os.path.exists(data_file):
            st.error(f"Data file '{data_file}' not found!")
            return None, False
        
        with st.spinner("Building vector store from reviews... This may take a minute."):
            try:
                raw_reviews = parse_reviews(data_file)
                reviews = prepare_reviews_for_rag(raw_reviews)
                pipeline.build_vectorstore(reviews, save_path=vectorstore_path)
                return pipeline, True
            except Exception as e:
                st.error(f"Error building vector store: {e}")
                return None, False

def display_review_result(result, _index):
    """Display a single review result in a nice format."""
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            rating = result['metadata'].get('rating', 'N/A')
            if rating == '5':
                st.markdown("⭐⭐⭐⭐⭐")
            elif rating == '4':
                st.markdown("⭐⭐⭐⭐")
            elif rating == '3':
                st.markdown("⭐⭐⭐")
            elif rating == '2':
                st.markdown("⭐⭐")
            elif rating == '1':
                st.markdown("⭐")
            else:
                st.markdown(f"**{rating}**")
            
            st.metric("Similarity", f"{1 - result['similarity_score']:.2%}")
        
        with col2:
            content = result['content']
            # Extract title if present
            if "Title:" in content:
                parts = content.split("\n\n", 1)
                if len(parts) > 1:
                    title = parts[0].replace("Title:", "").strip()
                    body = parts[1]
                    st.markdown(f"### {title}")
                    st.markdown(body)
                else:
                    st.markdown(content)
            else:
                st.markdown(content)
            
            # Show metadata
            with st.expander("View Metadata"):
                st.json(result['metadata'])
        
        st.divider()

def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Review RAG Dashboard</h1>', unsafe_allow_html=True)
    
    # Get Gemini settings from sidebar (will be set later)
    use_gemini_setting = st.session_state.get('use_gemini', True)
    
    # Load pipeline
    pipeline, success = load_pipeline(use_gemini=use_gemini_setting)
    
    if not success or pipeline is None:
        st.error("Failed to initialize the RAG pipeline. Please check the error messages above.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.radio(
            "Select Mode",
            ["Search", "Summarize", "Statistics", "Acceptance Tests"],
            help="Choose how you want to interact with the reviews"
        )
        
        num_results = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of reviews to retrieve"
        )
        
        use_gemini = st.checkbox(
            "Use Gemini for Summarization",
            value=True,
            help="Enable Google Gemini AI summarization (API key from config.py)"
        )
        st.session_state['use_gemini'] = use_gemini
        
        if use_gemini:
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-pro")
            if gemini_key:
                st.info(f"✅ Gemini API key configured: {gemini_key[:8]}...")
                st.info(f"📱 Model: {gemini_model}")
            else:
                st.warning("⚠️ Gemini API key not found. Check config.py")
        
        gemini_model_input = st.text_input(
            "Gemini Model (Optional)",
            value=os.getenv("GEMINI_MODEL", "gemini-pro"),
            help="Gemini model name (e.g., gemini-pro, gemini-pro-vision)"
        )
        
        if gemini_model_input:
            os.environ["GEMINI_MODEL"] = gemini_model_input
        
        st.divider()
        st.markdown("### 📈 Pipeline Status")
        st.success("✅ Pipeline Loaded")
        st.info("Vector store is ready for queries")
        
        st.divider()
        st.markdown("### 📊 Performance Metrics")
        if 'evaluator' not in st.session_state:
            st.session_state.evaluator = RAGEvaluator()
        
        if st.session_state.evaluator.metrics_history:
            stats = st.session_state.evaluator.get_summary_stats()
            st.metric("Avg Precision", f"{stats.get('avg_precision', 0):.2%}")
            st.metric("Avg Similarity", f"{stats.get('avg_similarity', 0):.2%}")
            st.metric("Total Queries", stats.get('total_queries', 0))
    
    # Main content area
    if mode == "Search":
        st.header("🔍 Search Reviews")
        st.markdown("Enter a query to find relevant customer reviews")
        
        query = st.text_input(
            "Search Query",
            placeholder="e.g., battery problems, overheating, display issues...",
            key="search_query"
        )
        
        if st.button("🔎 Search", type="primary"):
            if query:
                with st.spinner("Searching reviews..."):
                    try:
                        start_time = time.time()
                        results = pipeline.search_reviews(query, k=num_results)
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        # Calculate metrics
                        metrics = st.session_state.evaluator.evaluate_retrieval(
                            results, query, response_time
                        )
                        
                        if results:
                            st.success(f"Found {len(results)} relevant reviews")
                            
                            # Display KPIs
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Retrieval Precision",
                                    f"{metrics.retrieval_precision:.2%}",
                                    help="Percentage of retrieved documents that are relevant. Higher is better."
                                )
                            with col2:
                                st.metric(
                                    "Semantic Similarity",
                                    f"{metrics.semantic_similarity_score:.2%}",
                                    help="Average semantic similarity to query. Higher indicates better match."
                                )
                            with col3:
                                st.metric(
                                    "Response Time",
                                    f"{response_time:.3f}s",
                                    help="Time taken to retrieve results."
                                )
                            
                            # KPI Interpretation
                            with st.expander("📊 KPI Interpretation", expanded=False):
                                st.markdown("""
                                **Retrieval Precision** ({:.1f}%):
                                - Measures how many retrieved documents are actually relevant
                                - **Why it matters**: High precision means less noise in results, leading to better summaries
                                - **Interpretation**: 
                                    - >70%: Excellent - Most retrieved docs are relevant
                                    - 50-70%: Good - Majority are relevant
                                    - <50%: Needs improvement - Many irrelevant docs
                                
                                **Semantic Similarity** ({:.1f}%):
                                - Measures how semantically close retrieved docs are to your query
                                - **Why it matters**: Higher similarity means better semantic understanding by the embedding model
                                - **Interpretation**:
                                    - >60%: Excellent semantic match
                                    - 40-60%: Good match
                                    - <40%: Poor match, may need query refinement
                                """.format(
                                    metrics.retrieval_precision * 100,
                                    metrics.semantic_similarity_score * 100
                                ))
                            
                            # Display results
                            for i, result in enumerate(results, 1):
                                display_review_result(result, i)
                        else:
                            st.warning("No results found. Try a different query.")
                    except Exception as e:
                        st.error(f"Error during search: {e}")
            else:
                st.warning("Please enter a search query")
    
    elif mode == "Summarize":
        st.header("📝 Summarize Reviews")
        st.markdown("Get AI-powered summaries of reviews based on your query")
        
        query = st.text_input(
            "Summarization Query",
            placeholder="e.g., overall customer satisfaction, common issues, product quality...",
            key="summarize_query"
        )
        
        if st.button("📊 Generate Summary", type="primary"):
            if query:
                with st.spinner("Generating summary with Gemini..."):
                    try:
                        # Ensure pipeline uses Gemini if enabled
                        if st.session_state.get('use_gemini', True):
                            pipeline.use_gemini = True
                        summary = pipeline.summarize_reviews(query, k=num_results)
                        
                        st.success("Summary Generated")
                        st.markdown("### Summary")
                        st.markdown(summary)
                        
                        # Show source reviews count
                        with st.expander("View Source Information"):
                            st.info(f"Summary based on top {num_results} most relevant reviews")
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")
            else:
                st.warning("Please enter a query for summarization")
    
    elif mode == "Statistics":
        st.header("📊 Review Statistics & Performance Metrics")
        
        # Show KPI Performance
        if st.session_state.evaluator.metrics_history:
            st.subheader("RAG Performance KPIs")
            stats = st.session_state.evaluator.get_summary_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Precision", f"{stats.get('avg_precision', 0):.2%}")
                st.caption(f"Range: {stats.get('min_precision', 0):.2%} - {stats.get('max_precision', 0):.2%}")
            with col2:
                st.metric("Average Similarity", f"{stats.get('avg_similarity', 0):.2%}")
                st.caption(f"Range: {stats.get('min_similarity', 0):.2%} - {stats.get('max_similarity', 0):.2%}")
            with col3:
                st.metric("Avg Response Time", f"{stats.get('avg_time', 0):.3f}s")
                st.caption(f"Total Queries: {stats.get('total_queries', 0)}")
            
            st.markdown("---")
            st.markdown("### KPI Explanations")
            st.markdown("""
            **1. Retrieval Precision**
            - **What it measures**: Percentage of retrieved documents that are actually relevant to the query
            - **Why it matters**: High precision means the system is effectively filtering relevant reviews,
              reducing noise in the context passed to the LLM. This directly impacts answer quality and
              reduces hallucination.
            - **What the numbers mean**:
              - **>70%**: Excellent - System is retrieving mostly relevant documents
              - **50-70%**: Good - Majority of documents are relevant
              - **<50%**: Needs improvement - Many irrelevant documents being retrieved
            
            **2. Semantic Similarity Score**
            - **What it measures**: How semantically close retrieved documents are to the query
            - **Why it matters**: Higher scores indicate better semantic understanding by the embedding model.
              This KPI helps identify if the embedding model and vector search are working effectively.
            - **What the numbers mean**:
              - **>60%**: Excellent semantic match - Documents are highly relevant
              - **40-60%**: Good match - Documents are relevant
              - **<40%**: Poor match - Documents may not be semantically related to query
            """)
        
        st.markdown("---")
        
        # Original statistics
        st.subheader("Review Dataset Statistics")
        
        # Load reviews for statistics
        data_file = "Dell RAG compressed.txt"
        if os.path.exists(data_file):
            with st.spinner("Loading review statistics..."):
                try:
                    raw_reviews = parse_reviews(data_file)
                    reviews = prepare_reviews_for_rag(raw_reviews)
                    
                    # Calculate statistics
                    total_reviews = len(reviews)
                    ratings = [r['metadata'].get('rating', 'N/A') for r in reviews]
                    
                    # Rating distribution
                    rating_counts = {}
                    for rating in ratings:
                        rating_counts[rating] = rating_counts.get(rating, 0) + 1
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Reviews", total_reviews)
                    
                    with col2:
                        avg_rating = sum(int(r) for r in ratings if r.isdigit()) / len([r for r in ratings if r.isdigit()]) if any(r.isdigit() for r in ratings) else 0
                        st.metric("Average Rating", f"{avg_rating:.1f} ⭐")
                    
                    with col3:
                        five_star = rating_counts.get('5', 0)
                        st.metric("5-Star Reviews", five_star)
                    
                    with col4:
                        one_star = rating_counts.get('1', 0)
                        st.metric("1-Star Reviews", one_star)
                    
                    # Rating distribution chart
                    st.subheader("Rating Distribution")
                    st.bar_chart(rating_counts)
                    
                    # Top products (if available)
                    products = {}
                    for review in reviews:
                        product = review.get('product_name', 'Unknown')
                        if product:
                            products[product] = products.get(product, 0) + 1
                    
                    if products:
                        st.subheader("Reviews by Product")
                        st.bar_chart(products)
                    
                except Exception as e:
                    st.error(f"Error loading statistics: {e}")
        else:
            st.error("Data file not found for statistics")
    
    elif mode == "Acceptance Tests":
        st.header("🧪 Acceptance Tests")
        st.markdown("""
        Run acceptance tests to validate KPI calculations and system behavior.
        These tests verify that the RAG pipeline performs correctly in both normal and edge cases.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            run_happy = st.button("✅ Run Happy Path Test", type="primary", use_container_width=True)
        
        with col2:
            run_edge = st.button("⚠️ Run Edge Case Test", use_container_width=True)
        
        with col3:
            run_all = st.button("🔄 Run All Tests", use_container_width=True)
        
        if run_happy or run_all:
            st.markdown("---")
            st.subheader("📊 Happy Path Test")
            st.info("**Test Query**: 'battery problems' | **Expected**: High precision (>60%) and similarity (>45%)")
            
            with st.spinner("Running happy path test..."):
                # Capture output
                f = StringIO()
                with redirect_stdout(f):
                    try:
                        result = test_happy_path()
                        output = f.getvalue()
                    except Exception as e:
                        output = f.getvalue() + f"\nError: {e}"
                        result = False
                
                # Display results
                st.code(output, language=None)
                
                if result:
                    st.success("✅ **Test PASSED**: System correctly retrieves relevant documents with good KPIs")
                else:
                    st.warning("⚠️ **Test FAILED**: KPIs below expected thresholds")
        
        if run_edge or run_all:
            st.markdown("---")
            st.subheader("⚠️ Edge Case Test")
            st.info("**Test Query**: 'quantum computing performance' | **Expected**: Low precision (<40%) and similarity (<40%)")
            
            with st.spinner("Running edge case test..."):
                # Capture output
                f = StringIO()
                with redirect_stdout(f):
                    try:
                        result = test_edge_case()
                        output = f.getvalue()
                    except Exception as e:
                        output = f.getvalue() + f"\nError: {e}"
                        result = False
                
                # Display results
                st.code(output, language=None)
                
                if result:
                    st.success("✅ **Test PASSED**: System correctly identifies poor matches with low KPIs")
                else:
                    st.warning("⚠️ **Test FAILED**: KPIs not reflecting poor match quality as expected")
        
        if run_all:
            st.markdown("---")
            st.subheader("🔍 Empty Query Test")
            st.info("**Test Query**: 'a' (single character) | **Expected**: System handles gracefully")
            
            with st.spinner("Running empty query test..."):
                # Capture output
                f = StringIO()
                with redirect_stdout(f):
                    try:
                        result = test_empty_query()
                        output = f.getvalue()
                    except Exception as e:
                        output = f.getvalue() + f"\nError: {e}"
                        result = False
                
                # Display results
                st.code(output, language=None)
                
                if result:
                    st.success("✅ **Test PASSED**: System handles edge case gracefully")
                else:
                    st.error("❌ **Test FAILED**: System error on edge case")
            
            st.markdown("---")
            st.subheader("📋 Test Summary")
            st.markdown("""
            **What These Tests Validate:**
            
            1. **Happy Path**: Confirms the system works correctly for normal, well-formed queries
            2. **Edge Case**: Validates that KPIs correctly identify poor matches and the system handles out-of-scope queries gracefully
            3. **Empty Query**: Ensures the system doesn't crash on minimal input
            
            **Key Insights:**
            - KPIs should be high for relevant queries (happy path)
            - KPIs should be low for irrelevant queries (edge case)
            - System should always return results without crashing (graceful degradation)
            """)
        
        if not (run_happy or run_edge or run_all):
            st.info("👆 Click a button above to run acceptance tests")
            st.markdown("""
            ### About Acceptance Tests
            
            Acceptance tests validate that:
            - **KPI calculations are accurate**
            - **System behaves correctly in normal scenarios** (happy path)
            - **System handles edge cases gracefully** (failure cases)
            - **Expected vs actual results match** (showing understanding of the problem)
            
            Each test shows:
            - Expected behavior
            - Actual results
            - KPI values
            - Interpretation of what the numbers mean
            - Pass/Fail status
            """)
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Customer Review RAG Pipeline Dashboard | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

