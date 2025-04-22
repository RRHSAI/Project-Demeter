#!/usr/bin/env python3
"""
Advanced PDF Text Extraction and Cleaning Script

This script uses PyMuPDF's layout analysis and NLP techniques to extract
meaningful content from PDFs while filtering out boilerplate text,
headers, footers, and other irrelevant content for RAG applications.
"""

import os
import argparse
import re
import fitz  # PyMuPDF
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_blocks_from_pdf(pdf_path):
    """
    Extract text blocks with position information using PyMuPDF's layout analysis.
    Returns text blocks with their positions and page numbers.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Get blocks which contain (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")
        
        for block in blocks:
            # Only keep text blocks (ignore images etc.)
            if block[6] == 0:  # Text blocks have type 0
                x0, y0, x1, y1 = block[:4]
                text = block[4]
                
                # Calculate normalized position (useful for detecting headers/footers)
                # Normalize positions to page dimensions
                page_width = page.rect.width
                page_height = page.rect.height
                
                norm_position = {
                    "left": x0 / page_width,
                    "top": y0 / page_height,
                    "right": x1 / page_width,
                    "bottom": y1 / page_height,
                    "width": (x1 - x0) / page_width,
                    "height": (y1 - y0) / page_height,
                    "x_center": (x0 + x1) / (2 * page_width),
                    "y_center": (y0 + y1) / (2 * page_height)
                }
                
                # Store block data with position info and page number
                all_blocks.append({
                    "text": text.strip(),
                    "position": norm_position,
                    "page": page_num,
                    "raw_coords": (x0, y0, x1, y1)
                })
    
    doc.close()
    return all_blocks


def identify_headers_and_footers(blocks, threshold=0.7):
    """
    Identify headers and footers based on position and repetition patterns.
    Returns a list of block indices that are likely headers or footers.
    """
    # Extract features for positioning
    features = []
    for block in blocks:
        pos = block["position"]
        # Use y-position as the main feature to cluster headers/footers
        features.append([pos["y_center"]])
    
    features = np.array(features)
    
    # Use DBSCAN to cluster blocks by position
    clustering = DBSCAN(eps=0.03, min_samples=2).fit(features)
    labels = clustering.labels_
    
    # Find clusters that appear on multiple pages (likely headers/footers)
    header_footer_indices = []
    
    # Group blocks by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # For each cluster, check if it appears on multiple pages
    for label, indices in clusters.items():
        # Get pages for this cluster
        pages = [blocks[i]["page"] for i in indices]
        unique_pages = len(set(pages))
        
        # If cluster appears on multiple pages and is near top or bottom, it's likely a header/footer
        if unique_pages >= max(3, len(pages) * threshold):
            # Check if all blocks in this cluster are at similar positions
            y_positions = [blocks[i]["position"]["y_center"] for i in indices]
            y_mean = np.mean(y_positions)
            
            # Headers are typically at the top, footers at the bottom
            if y_mean < 0.2 or y_mean > 0.8:
                header_footer_indices.extend(indices)
    
    return header_footer_indices


def detect_boilerplate_text(blocks):
    """
    Detect boilerplate text by analyzing text similarity across blocks.
    Returns a list of block indices that are likely boilerplate.
    """
    texts = [block["text"] for block in blocks]
    
    # Use TF-IDF to compare text similarity
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.5)
    
    # Handle case with too few blocks
    if len(texts) < 3:
        return []
    
    try:
        X = vectorizer.fit_transform(texts)
        
        # Calculate similarity matrix
        similarity = (X * X.T).toarray()
        np.fill_diagonal(similarity, 0)  # Zero out self-similarity
        
        # Find blocks with high similarity to many others
        boilerplate_indices = []
        for i, row in enumerate(similarity):
            # If a block is similar to more than 40% of other blocks, it's likely boilerplate
            if sum(row > 0.8) > max(3, len(texts) * 0.2):
                boilerplate_indices.append(i)
        
        return boilerplate_indices
    except:
        # If vectorization fails, return empty list
        return []


def is_navigational_text(text):
    """Detect navigational elements like page numbers, TOC references, etc."""
    # Check for page numbers and references
    if re.match(r'^Page \d+( of \d+)?$', text.strip()):
        return True
    
    # Check for navigation elements
    nav_patterns = [
        r'^Table of Contents$',
        r'^Index$',
        r'^Chapter \d+$',
        r'^Section \d+\.\d+$',
        r'^\d+\.\s+.{1,50}\.{3,}\d+$',  # TOC entries with page numbers
        r'^See page \d+$',
    ]
    
    for pattern in nav_patterns:
        if re.match(pattern, text.strip()):
            return True
    
    return False


def looks_like_legal_boilerplate(text):
    """Detect legal boilerplate text such as copyright notices, disclaimers, etc."""
    text = text.lower()
    
    # Check for common legal terms in combination
    legal_indicators = [
        'copyright', 'all rights reserved', 'trademark', 'confidential',
        'proprietary', 'legal notice', 'disclaimer', 'terms of use',
        'terms and conditions', 'privacy policy', 'warranty'
    ]
    
    # Count how many legal indicators appear in the text
    indicator_count = sum(1 for term in legal_indicators if term in text)
    
    # If multiple indicators are present, it's likely legal boilerplate
    return indicator_count >= 2


def clean_and_filter_blocks(blocks):
    """
    Clean and filter blocks to remove headers, footers, boilerplate, and other junk.
    Returns a list of cleaned, filtered blocks.
    """
    # Identify blocks to remove
    header_footer_indices = identify_headers_and_footers(blocks)
    boilerplate_indices = detect_boilerplate_text(blocks)
    
    # Combine all indices to remove
    indices_to_remove = set(header_footer_indices + boilerplate_indices)
    
    # Filter and clean blocks
    cleaned_blocks = []
    for i, block in enumerate(blocks):
        # Skip if block is identified as header/footer or boilerplate
        if i in indices_to_remove:
            continue
        
        text = block["text"]
        
        # Skip if block is empty or too short
        if not text or len(text.strip()) < 4:
            continue
            
        # Skip navigational elements
        if is_navigational_text(text):
            continue
            
        # Skip legal boilerplate
        if looks_like_legal_boilerplate(text):
            continue
        
        # Clean remaining text
        cleaned_text = text
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # Add cleaned block to results
        if cleaned_text:
            cleaned_blocks.append({
                "text": cleaned_text,
                "page": block["page"]
            })
    
    return cleaned_blocks


def process_pdf(input_path, output_path=None, keep_page_breaks=False):
    """
    Process a PDF file and save the cleaned text.
    
    Args:
        input_path: Path to the input PDF file
        output_path: Path to save the cleaned text (default: input_name_cleaned.txt)
        keep_page_breaks: Whether to keep page break indicators in the output
    """
    # Extract text blocks with position info
    print(f"Extracting text from {input_path}...")
    blocks = extract_blocks_from_pdf(input_path)
    
    # Clean and filter blocks
    print("Analyzing and cleaning text...")
    cleaned_blocks = clean_and_filter_blocks(blocks)
    
    # Sort blocks by page and position (top to bottom)
    cleaned_blocks.sort(key=lambda b: (b["page"], b.get("position", {}).get("y_center", 0)))
    
    # Combine blocks into a single text
    current_page = -1
    cleaned_text = ""
    
    for block in cleaned_blocks:
        # Add page break if needed
        if keep_page_breaks and block["page"] != current_page and current_page != -1:
            cleaned_text += "\n\n--- Page Break ---\n\n"
        
        # Add block text
        cleaned_text += block["text"] + "\n\n"
        current_page = block["page"]
    
    # Determine output path
    if not output_path:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_cleaned.txt"
    
    # Save the cleaned text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"Cleaned text saved to {output_path}")
    print(f"Original block count: {len(blocks)}, Filtered block count: {len(cleaned_blocks)}")
    
    return cleaned_text


def main():
    parser = argparse.ArgumentParser(description="Extract and clean text from PDF files for RAG models")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", help="Path to save the cleaned text (default: input_name_cleaned.txt)")
    parser.add_argument("--keep-page-breaks", action="store_true", help="Include page break markers in the output")
    
    args = parser.parse_args()
    
    process_pdf(args.input_pdf, args.output, args.keep_page_breaks)


if __name__ == "__main__":
    # Local testing setup
    test_pdf_path = "data/your_test_file.pdf"  # Replace with your actual PDF path
    cleaned_blocks = clean_and_filter_blocks(extract_blocks_from_pdf(test_pdf_path))

    print(f"\n🔍 Total Cleaned Blocks: {len(cleaned_blocks)}\n")

    for i, block in enumerate(cleaned_blocks[:10]):  # Show first 10 blocks
        print(f"--- Block {i + 1} ---")
        print(f"Page: {block['page']}")
        print(f"Text:\n{block['text']}\n")
        print("-" * 40)