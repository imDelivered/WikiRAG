"""Article chunking strategy for RAG system."""

import re
from typing import List

from kiwix_chat.models import Chunk


def chunk_article(
    text: str,
    title: str,
    href: str,
    max_chunk_size: int = 500,
    overlap: int = 50,
    content_type: str = "Kiwix content"
) -> List[Chunk]:
    """Split article text into chunks with overlap.
    
    Strategy:
    - Split by sentences, respect paragraph boundaries
    - Preserve article metadata in each chunk
    - Use overlap to maintain context between chunks
    
    Args:
        text: Full article text
        title: Article title
        href: Article href/path
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        content_type: Type of content (e.g., "Wikipedia articles")
        
    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []
    
    # Clean text: normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # If text is small enough, return single chunk
    if len(text) <= max_chunk_size:
        return [Chunk(
            text=text,
            article_title=title,
            href=href,
            chunk_idx=0,
            total_chunks=1,
            content_type=content_type
        )]
    
    # Split into sentences (preserve sentence boundaries)
    # Pattern: sentence ending followed by space and capital letter or end of text
    sentences = re.split(r'([.!?]+\s+)', text)
    
    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    # Build chunks from sentences
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in combined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_size = len(sentence)
        
        # If single sentence exceeds max_chunk_size, split it by words
        if sentence_size > max_chunk_size:
            # First, save current chunk if it has content
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    article_title=title,
                    href=href,
                    chunk_idx=len(chunks),
                    total_chunks=0,  # Will update later
                    content_type=content_type
                ))
                current_chunk = []
                current_size = 0
            
            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_size = 0
            
            for word in words:
                word_len = len(word) + 1  # +1 for space
                if word_size + word_len > max_chunk_size and word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        article_title=title,
                        href=href,
                        chunk_idx=len(chunks),
                        total_chunks=0,
                        content_type=content_type
                    ))
                    # Start new chunk with overlap
                    overlap_words = word_chunk[-overlap//10:] if len(word_chunk) > overlap//10 else word_chunk
                    word_chunk = overlap_words + [word]
                    word_size = sum(len(w) + 1 for w in word_chunk)
                else:
                    word_chunk.append(word)
                    word_size += word_len
            
            # Add remaining words
            if word_chunk:
                chunk_text = ' '.join(word_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    article_title=title,
                    href=href,
                    chunk_idx=len(chunks),
                    total_chunks=0,
                    content_type=content_type
                ))
            
            continue
        
        # Check if adding this sentence would exceed max_chunk_size
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                article_title=title,
                href=href,
                chunk_idx=len(chunks),
                total_chunks=0,  # Will update later
                content_type=content_type
            ))
            
            # Start new chunk with overlap (last N characters from previous chunk)
            if overlap > 0 and len(chunk_text) > overlap:
                overlap_text = chunk_text[-overlap:]
                # Try to start at word boundary
                overlap_start = overlap_text.find(' ')
                if overlap_start > 0:
                    overlap_text = overlap_text[overlap_start + 1:]
                current_chunk = [overlap_text, sentence]
                current_size = len(overlap_text) + sentence_size + 1
            else:
                current_chunk = [sentence]
                current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(Chunk(
            text=chunk_text,
            article_title=title,
            href=href,
            chunk_idx=len(chunks),
            total_chunks=0,
            content_type=content_type
        ))
    
    # Update total_chunks for all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total
    
    return chunks

