#!/usr/bin/env python3
"""
3GPP Specification Ingestion & Vector Indexing Pipeline - ASCII Safe
===================================================================
Reads raw spec documents (.docx, .pdf, .txt) from the 3gpp_docs folder,
chunks them recursively, generates embeddings via sentence-transformers,
and serializes a search index (FAISS index or Numpy fallback matrix).
"""

import os
import sys
import re
import pickle
import argparse
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import List, Dict, Tuple

# Simple helper mapping to assign descriptive titles to specs based on IDs
SPEC_TITLES = {
    "TS_38.331": "NR; Radio Resource Control (RRC); Protocol specification",
    "TS_23.501": "System architecture for the 5G System (5GS)",
    "TS_24.501": "Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
    "TS_38.300": "NR; NR and NG-RAN Overall Description",
    "TS_38.401": "NG-RAN; Architecture description",
    "TS_24.229": "IP multimedia call control protocol based on SIP and SDP",
    "TS_38.213": "NR; Physical layer procedures for control",
    "TS_38.214": "NR; Physical layer procedures for data",
    "TS_38.321": "NR; Medium Access Control (MAC) protocol specification",
    "TS_38.304": "NR; User Equipment (UE) procedures in Idle mode"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest 3GPP specs and build RAG index.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3gpp_docs"),
        help="Directory containing raw document files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index"),
        help="Directory to save the generated FAISS index and metadata."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=850,
        help="Target length of text chunks in characters."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Overlap size between contiguous text chunks."
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution for SentenceTransformer embeddings."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk documents only, without generating embeddings or saving index files."
    )
    return parser.parse_args()

def extract_text_from_txt(file_path: str) -> str:
    """Reads standard text files."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"  [WARNING] Error reading txt file '{file_path}': {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """Reads PDF files using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text_parts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        print("  [WARNING] 'pypdf' library is missing. Cannot parse PDF files. Install with 'pip install pypdf'.")
        return ""
    except Exception as e:
        print(f"  [WARNING] Error reading PDF file '{file_path}': {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """
    Reads Word (.docx) documents using standard python libraries.
    Extracts text blocks from word/document.xml to avoid heavy native library dependencies.
    """
    try:
        paragraphs = []
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Find all paragraph tags <w:p>
            for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                # Extract text segments from text tags <w:t> inside this paragraph
                texts = [node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append("".join(texts))
        return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"  [WARNING] Error reading DOCX file '{file_path}' (zip/xml extraction): {e}")
        return ""

def recursive_chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text recursively to create overlapping chunks.
    Attempts to preserve paragraph boundaries, falling back to sentences for large paragraphs.
    """
    if not text or len(text.strip()) == 0:
        return []
        
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_len = len(para)
        
        # If paragraph exceeds chunk size, split it by sentence boundaries
        if para_len > chunk_size:
            # Drain current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            # Split by sentences (handling standard punctuation boundaries)
            sentences = re.split(r'(?<=[.?!])\s+', para)
            sub_chunk = []
            sub_len = 0
            for sent in sentences:
                sent_len = len(sent)
                if sub_len + sent_len > chunk_size:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                    sub_chunk = [sent]
                    sub_len = sent_len
                else:
                    sub_chunk.append(sent)
                    sub_len += sent_len + 1
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        else:
            if current_length + para_len > chunk_size:
                chunks.append("\n\n".join(current_chunk))
                
                # Capture overlap from the end of the current chunk
                overlap_chars = 0
                overlap_chunk = []
                for prev_para in reversed(current_chunk):
                    if overlap_chars + len(prev_para) <= chunk_overlap:
                        overlap_chunk.insert(0, prev_para)
                        overlap_chars += len(prev_para) + 2
                    else:
                        break
                current_chunk = overlap_chunk + [para]
                current_length = sum(len(p) for p in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk.append(para)
                current_length += para_len + 2
                
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def extract_section_label(chunk_text: str) -> str:
    """Attempts to identify the section header inside a chunk of text."""
    # Search for headings like "5.3.5", "Section 4.1", "4.1.2.3"
    lines = [l.strip() for l in chunk_text.split('\n') if l.strip()]
    
    # Check the first few lines of the chunk
    for line in lines[:3]:
        # Regex matching sections starting with numbers (e.g. "5.1.2.3  Random Access")
        match = re.match(r'^(\d+\.\d+(?:\.\d+)*)\s+([A-Za-z].*)$', line)
        if match:
            return f"Section {match.group(1)}: {match.group(2)[:40]}"
            
        # Regex matching "Section X.Y"
        match_sec = re.match(r'^(Section\s+\d+\.\d+)', line, re.IGNORECASE)
        if match_sec:
            return line[:50]
            
    # Fallback to checking line regex matches in the whole chunk
    for line in lines:
        match = re.match(r'^(\d+\.\d+(?:\.\d+)*)\s+([A-Za-z].*)$', line)
        if match:
            return f"Section {match.group(1)}: {match.group(2)[:40]}"
            
    return "General Content"

def main():
    args = parse_args()
    
    print("=" * 70)
    print("[INFO] 3GPP Document Ingestion & Indexing Engine")
    print("=" * 70)
    print(f"Input Directory:  {os.path.abspath(args.input_dir)}")
    print(f"Output Directory: {os.path.abspath(args.output_dir)}")
    print(f"Chunk Configuration: Size={args.chunk_size} chars, Overlap={args.chunk_overlap} chars")
    print("=" * 70)
    
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
        
    # 1. Scan for documents
    valid_extensions = ('.txt', '.pdf', '.docx')
    doc_files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.lower().endswith(valid_extensions)
    ]
    
    if not doc_files:
        print(f"[WARNING] No valid document files (.txt, .pdf, .docx) found in '{args.input_dir}'.")
        print("Please run 'python -m rag_pipeline.download_specs' first to fetch sample specs.")
        sys.exit(0)
        
    print(f"Found {len(doc_files)} files to index.")
    
    # 2. Extract and chunk text
    all_chunks = []
    
    for doc_path in doc_files:
        filename = os.path.basename(doc_path)
        print(f"\nParsing {filename}...")
        
        # Determine Spec ID and Title
        base_name = os.path.splitext(filename)[0]
        # Match pattern like "TS_38.331" or "TS_38_331" or just "38.331"
        spec_match = re.search(r'(TS_?\d{2}\.?\d{3})', base_name)
        spec_id = spec_match.group(1).replace('_', ' ') if spec_match else "TS Reference"
        
        # Lookup clean title or map generic one
        spec_key = spec_id.replace(' ', '_')
        spec_title = SPEC_TITLES.get(spec_key, "3GPP Telecom Specification")
        
        # Read file content
        ext = os.path.splitext(doc_path)[1].lower()
        if ext == '.txt':
            raw_text = extract_text_from_txt(doc_path)
        elif ext == '.pdf':
            raw_text = extract_text_from_pdf(doc_path)
        elif ext == '.docx':
            raw_text = extract_text_from_docx(doc_path)
        else:
            raw_text = ""
            
        if not raw_text or len(raw_text.strip()) == 0:
            print(f"  [WARNING] Skipped: No readable text extracted from {filename}.")
            continue
            
        print(f"  Extracted {len(raw_text)} characters.")
        
        # Split into chunks
        chunks = recursive_chunk_text(raw_text, args.chunk_size, args.chunk_overlap)
        print(f"  Generated {len(chunks)} text chunks.")
        
        # Add metadata tagging
        for chunk in chunks:
            if len(chunk.strip()) < 50:  # Skip tiny fragments
                continue
                
            section = extract_section_label(chunk)
            
            # Simple keyword extraction to match existing structure
            words = set(re.findall(r'[a-zA-Z]{4,}', chunk.lower()))
            # Pick up to 5 longest words as keywords if none are specified
            keywords = sorted(list(words), key=len, reverse=True)[:5]
            
            all_chunks.append({
                "spec_id": spec_id,
                "title": spec_title,
                "section": section,
                "content": chunk,
                "keywords": keywords
            })
            
    if not all_chunks:
        print("[ERROR] No text chunks extracted. Vector database index aborted.")
        sys.exit(1)
        
    print(f"\n[INFO] Extracted {len(all_chunks)} total chunks across all documents.")
    
    if args.dry_run:
        print("\n" + "=" * 70)
        print("[SUCCESS] Dry Run Complete!")
        print("Parsed and chunked documents successfully without generating embeddings.")
        if all_chunks:
            print(f"Total Chunks: {len(all_chunks)}")
            print(f"First Chunk Spec ID: {all_chunks[0]['spec_id']}")
            print(f"First Chunk Section: {all_chunks[0]['section']}")
            print(f"First Chunk Preview (first 250 chars):\n--------------------------------------------------\n{all_chunks[0]['content'][:250]}...\n--------------------------------------------------")
        print("=" * 70)
        sys.exit(0)
    
    # 3. Generate Vector Embeddings
    print("\n[INFO] Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model hardware acceleration device: {device.upper()}")
        
        # Set cache dir to local folder to bypass network permissions issues
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(args.output_dir), "cache")
        embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    except ImportError:
        print("[ERROR] 'sentence-transformers' package is missing. Run: pip install sentence-transformers")
        sys.exit(1)
        
    print("Generating embeddings for all chunks (this may take a few minutes)...")
    start_time = time.time()
    
    # Extract only content for encoding
    texts_to_embed = [f"{c['section']}: {c['content']}" for c in all_chunks]
    embeddings = embed_model.encode(texts_to_embed, show_progress_bar=True, batch_size=32)
    
    import numpy as np
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"[SUCCESS] Embeddings generated in {time.time() - start_time:.2f} seconds.")
    print(f"Embedding dimensions: {embeddings.shape}")
    
    # 4. Save Index and Metadata
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "metadata.pkl")
    
    # Try compiling FAISS Index
    faiss_saved = False
    try:
        import faiss
        dim = embeddings.shape[1]
        
        # Normalize vectors for cosine similarity (Inner Product flat index)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        faiss_path = os.path.join(args.output_dir, "index.faiss")
        faiss.write_index(index, faiss_path)
        print(f"[SUCCESS] FAISS index saved successfully: {faiss_path}")
        faiss_saved = True
    except ImportError:
        print("[WARNING] FAISS is not installed. Saving Numpy fallback vector matrix...")
        
    # Numpy Fallback Save
    if not faiss_saved:
        numpy_path = os.path.join(args.output_dir, "index.npy")
        np.save(numpy_path, embeddings)
        print(f"[SUCCESS] Numpy vector matrix saved successfully: {numpy_path}")
        
    # Save Metadata Chunks list
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"[SUCCESS] Document metadata saved successfully: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Ingestion Pipeline Complete!")
    print(f"Successfully processed {len(doc_files)} specs and indexed {len(all_chunks)} chunks.")
    print(f"Index folder: {os.path.abspath(args.output_dir)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
