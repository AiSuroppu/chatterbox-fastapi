import re
import logging
import pysbd
from typing import List
from functools import lru_cache

@lru_cache(maxsize=2)
def get_segmenter(language: str) -> pysbd.Segmenter:
    """
    Returns a cached instance of a pysbd.Segmenter for a given language.
    This prevents re-initializing the segmenter on every request.
    """
    logging.info(f"Initializing pysbd.Segmenter for language: '{language}'")
    return pysbd.Segmenter(language=language, clean=False)

def clean_text(text: str, remove_brackets: bool) -> str:
    """Performs a series of cleaning operations on the input text."""
    # Normalize different dash types to a standard hyphen
    text = re.sub(r'[–—]', '-', text)
    # Normalize smart quotes to standard quotes
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    
    if remove_brackets:
        # Remove text in brackets, parentheses, or asterisks (often for stage directions)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\*.*?\*', '', text)
        
    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _force_split_long_chunk(chunk: str, max_length: int) -> List[str]:
    """
    Intelligently splits a chunk that is longer than max_length.
    It prioritizes splitting at punctuation, then spaces, as a last resort.
    """
    if len(chunk) <= max_length:
        return [chunk]
    
    # Prioritized list of characters to split on, from most to least desirable
    split_chars = [';', ':', ',', ' ']
    
    best_split_pos = -1
    for char in split_chars:
        pos = chunk.rfind(char, 0, max_length)
        if pos > best_split_pos:
            best_split_pos = pos

    if best_split_pos == -1:
        # No preferred character found, so we hard-cut at max_length
        best_split_pos = max_length
        
    first_part = chunk[:best_split_pos].strip()
    second_part = chunk[best_split_pos:].strip()
    
    # Recursively split the rest of the chunk to handle very long inputs
    return [first_part] + _force_split_long_chunk(second_part, max_length)

def process_and_chunk_text(
    text: str, 
    text_options: 'TextProcessingOptions'
) -> list[str]:
    """
    Cleans and chunks text according to the specified strategy, ensuring no chunk
    exceeds the max_chunk_length.
    """
    if text_options.batching_strategy == 'simple':
        cleaned_text = clean_text(text, text_options.remove_bracketed_text)
        if text_options.to_lowercase:
            cleaned_text = cleaned_text.lower()
        return _force_split_long_chunk(cleaned_text, text_options.max_chunk_length)

    # 'recursive' strategy (default)
    segmenter = get_segmenter(text_options.text_language)
    initial_sentences = segmenter.segment(text)
    
    # Proactively split any sentence that is itself too long
    sentences = []
    for sent in initial_sentences:
        sent = clean_text(sent, text_options.remove_bracketed_text)
        if text_options.to_lowercase:
            sent = sent.lower()
        sentences.extend(_force_split_long_chunk(sent, text_options.max_chunk_length))

    # Group sentences into chunks that respect max_chunk_length
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if not current_chunk:
            current_chunk = sentence
        elif len(current_chunk) + len(sentence) + 1 <= text_options.max_chunk_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
        
    logging.info(f"Processed text into {len(chunks)} chunks using '{text_options.batching_strategy}' strategy.")
    return chunks