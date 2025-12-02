#!/bin/bash
# Quick start script for the RAG Dashboard

echo "🚀 Starting Customer Review RAG Dashboard..."
echo ""
echo "The dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
streamlit run dashboard.py

