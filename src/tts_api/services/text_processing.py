import re
import logging
import pysbd
from typing import List, Sequence
from functools import lru_cache

from tts_api.core.models import TextChunkingStrategy

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

def _force_split_long_chunk(
    chunk: str,
    max_length: int,
    split_delimiters: Sequence[str] = (".", "?", "!", ";", ":", ",", " ")
) -> List[str]:
    """
    Splits a text chunk into smaller parts, ensuring no part exceeds max_length.

    This function is 'intelligent' because it doesn't just cut the text at
    max_length. Instead, it searches backwards from the `max_length` point
    to find a natural break point based on a prioritized list of delimiters.

    The splitting logic is as follows:
    1. If the chunk is already short enough, it's returned as a single item list.
    2. Otherwise, it finds the best possible split point within the first
       `max_length` characters.
    3. The "best" split point is determined by iterating through `split_delimiters`
       and finding the right-most occurrence of the highest-priority delimiter.
    4. If no delimiter is found, it performs a "hard" cut at `max_length`.
    5. The process is repeated on the remainder of the string until the entire
       chunk has been processed.

    Args:
        chunk: The input string to be split.
        max_length: The maximum allowed length for any given sub-chunk.
        split_delimiters: A sequence of strings to split on, in order of
                          preference (most to least desirable).

    Returns:
        A list of strings, where each string is no longer than max_length.
    """
    if not chunk:
        return []
    if len(chunk) <= max_length:
        return [chunk]

    results = []
    current_pos = 0
    while current_pos < len(chunk):
        remaining_chunk = chunk[current_pos:]

        # If the rest of the chunk is short enough, we're done
        if len(remaining_chunk) <= max_length:
            results.append(remaining_chunk)
            break

        split_pos = -1

        # Define the search window: from the start of the remaining_chunk up to max_length
        # We search up to max_length + 1 to include a delimiter at exactly max_length
        search_window = remaining_chunk[:max_length + 1]

        # Find the best delimiter by priority
        for delimiter in split_delimiters:
            # Find the last occurrence of this delimiter in the window
            pos = search_window.rfind(delimiter)
            if pos != -1:
                # We found a delimiter. The split should happen *after* it.
                split_pos = pos + 1
                break  # Exit after finding the highest-priority delimiter

        # If no delimiter was found, perform a hard cut at max_length
        if split_pos == -1:
            split_pos = max_length
        
        # Extract the piece and add it to results
        piece_to_add = remaining_chunk[:split_pos].strip()
        if piece_to_add: # Avoid adding empty strings from repeated spaces/delimiters
            results.append(piece_to_add)
        
        # Move the main cursor forward
        current_pos += split_pos

    return results

def _prepare_sentences(text: str, options: 'TextProcessingOptions') -> List[str]:
    """Segments, cleans, and pre-splits overly long sentences."""
    segmenter = get_segmenter(options.text_language)
    processed_sentences = []
    for sent in segmenter.segment(text):
        sent = clean_text(sent, options.remove_bracketed_text).strip()
        if options.to_lowercase: sent = sent.lower()
        if not sent: continue
        if len(sent) > options.max_chunk_length:
            processed_sentences.extend(
                _force_split_long_chunk(sent, options.max_chunk_length)
            )
        else:
            processed_sentences.append(sent)
    return processed_sentences

def _calculate_chunk_cost(length: int, options: 'TextProcessingOptions') -> float:
    """Calculates the 'badness' of a chunk of a given length."""
    if length > options.max_chunk_length: return float('inf')
    cost = ((length - options.ideal_chunk_length) / options.ideal_chunk_length) ** 2
    if length < options.ideal_chunk_length * 0.5:
        cost *= options.shortness_penalty_factor
    return cost

def _chunk_sentences_balanced(sentences: List[str], options: 'TextProcessingOptions') -> List[str]:
    """Chunks sentences using dynamic programming for globally optimal splits."""
    if not sentences: return []
    n = len(sentences)
    sentence_lens = [len(s) for s in sentences]
    min_costs, split_points = [0.0] + [float('inf')] * n, [0] * (n + 1)

    for i in range(1, n + 1):
        current_chunk_len = -1  # Start at -1 to account for the first space being added
        for j in range(i, 0, -1):
            current_chunk_len += sentence_lens[j-1] + 1
            if current_chunk_len > options.max_chunk_length: break
            cost = _calculate_chunk_cost(current_chunk_len, options)
            total_cost = min_costs[j-1] + cost
            if total_cost < min_costs[i]:
                min_costs[i], split_points[i] = total_cost, j-1
    
    chunks, end = [], n
    while end > 0:
        start = split_points[end]
        chunks.append(" ".join(sentences[start:end]))
        end = start
    return chunks[::-1]

def _chunk_sentences_greedy(sentences: List[str], options: 'TextProcessingOptions') -> List[str]:
    """Groups sentences greedily. Fast but can create uneven chunks."""
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
        elif len(current_chunk) + len(sentence) + 1 <= options.max_chunk_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    return chunks

def process_and_chunk_text(text: str, options: 'TextProcessingOptions') -> List[str]:
    """
    Cleans and chunks text according to a specified strategy..
    """
    strategy = options.chunking_strategy
    
    if strategy == TextChunkingStrategy.PARAGRAPH:
        chunks = []
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            
            cleaned_para = clean_text(para, options.remove_bracketed_text)
            if options.to_lowercase: cleaned_para = cleaned_para.lower()
            if not cleaned_para: continue

            if len(cleaned_para) <= options.max_chunk_length:
                chunks.append(cleaned_para)
            else:
                sentences = _prepare_sentences(para, options)
                chunks.extend(_chunk_sentences_balanced(sentences, options))

    elif strategy == TextChunkingStrategy.BALANCED:
        sentences = _prepare_sentences(text, options)
        chunks = _chunk_sentences_balanced(sentences, options)
        
    elif strategy == TextChunkingStrategy.GREEDY:
        sentences = _prepare_sentences(text, options)
        chunks = _chunk_sentences_greedy(sentences, options)
        
    elif strategy == TextChunkingStrategy.SIMPLE:
        cleaned_text = clean_text(text, options.remove_bracketed_text)
        if options.to_lowercase: cleaned_text = cleaned_text.lower()
        chunks = _force_split_long_chunk(cleaned_text, options.max_chunk_length)
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'")
        
    final_chunks = [chunk for chunk in chunks if chunk]
    logging.info(f"Processed text into {len(final_chunks)} chunks using '{strategy}' strategy.")
    return final_chunks