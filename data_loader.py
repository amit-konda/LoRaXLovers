"""
Data loader for parsing customer reviews from the text file.
"""
import pandas as pd
from typing import List, Dict
import re


def parse_reviews(file_path: str) -> List[Dict]:
    """
    Parse reviews from the tab-separated text file.
    
    Args:
        file_path: Path to the reviews text file
        
    Returns:
        List of dictionaries containing review data
    """
    reviews = []
    
    try:
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                # Split by tab
                parts = line.strip().split('\t')
                
                if len(parts) < 3:
                    continue
                
                # Extract key information
                # Format appears to be: rating, title, body, product_id, reviewer_id, ...
                try:
                    rating = parts[0].strip()
                    title = parts[1].strip() if len(parts) > 1 else ""
                    body = parts[2].strip() if len(parts) > 2 else ""
                    
                    # Get product name if available (usually around index 7-8)
                    product_name = ""
                    if len(parts) > 7:
                        product_name = parts[7].strip()
                    
                    # Create review document
                    if body:  # Only include reviews with body text
                        review_text = f"Title: {title}\n\n{body}"
                        if product_name:
                            review_text = f"Product: {product_name}\n\n{review_text}"
                        
                        reviews.append({
                            'rating': rating,
                            'title': title,
                            'body': body,
                            'product_name': product_name,
                            'text': review_text,
                            'line_number': line_num
                        })
                except Exception as e:
                    # Skip malformed lines
                    continue
                    
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return reviews


def clean_text(text: str) -> str:
    """
    Clean review text by removing HTML tags and extra whitespace.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove video IDs
    text = re.sub(r'\[\[VIDEOID:[^\]]+\]\]', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def prepare_reviews_for_rag(reviews: List[Dict]) -> List[Dict]:
    """
    Prepare reviews for RAG by cleaning and formatting.
    
    Args:
        reviews: List of review dictionaries
        
    Returns:
        List of cleaned review dictionaries
    """
    prepared = []
    
    for review in reviews:
        cleaned_body = clean_text(review['body'])
        cleaned_title = clean_text(review['title'])
        
        if cleaned_body:  # Only include reviews with content
            prepared.append({
                'rating': review['rating'],
                'title': cleaned_title,
                'body': cleaned_body,
                'product_name': review.get('product_name', ''),
                'text': f"Rating: {review['rating']} stars\nTitle: {cleaned_title}\n\nReview: {cleaned_body}",
                'metadata': {
                    'rating': review['rating'],
                    'product_name': review.get('product_name', ''),
                    'line_number': review.get('line_number', 0)
                }
            })
    
    return prepared

