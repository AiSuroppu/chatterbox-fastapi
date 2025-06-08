import re
import logging
from typing import List, Sequence
from functools import lru_cache

import pysbd
from nemo_text_processing.text_normalization.normalize import Normalizer

from tts_api.core.config import settings
from tts_api.core.models import TextChunkingStrategy, TextProcessingOptions

@lru_cache(maxsize=settings.PYSBD_CACHE_SIZE)
def get_segmenter(language: str) -> pysbd.Segmenter:
    """
    Returns a cached instance of a pysbd.Segmenter for a given language.
    This prevents re-initializing the segmenter on every request.
    """
    logging.info(f"Initializing pysbd.Segmenter for language: '{language}'")
    # clean=False is important as we perform our own cleaning.
    return pysbd.Segmenter(language=language, clean=False)

@lru_cache(maxsize=settings.NEMO_NORMALIZER_CACHE_SIZE)
def get_normalizer(language: str) -> Normalizer:
    """
    Returns a cached instance of the NeMo Normalizer for a given language.
    This prevents re-initializing the complex grammar WFSTs on every request.
    """
    logging.info(f"Initializing NeMo Normalizer for language: '{language}'")
    # Using the configured cache directory. overwrite_cache=False is a performance
    # optimization to avoid re-compiling grammars that already exist in the cache.
    return Normalizer(
        input_case='cased',
        lang=language,
        deterministic=True,
        cache_dir=settings.NEMO_CACHE_DIR,
        overwrite_cache=False
    )

def _normalize_whitespace_and_unicode(text: str) -> str:
    """
    Performs foundational, non-destructive text cleaning.
    - Normalizes various unicode and control characters.
    - Collapses multiple whitespace characters into a single space.
    """
    normalization_rules = [
        # --- Quotes and Dashes ---
        #('’', "'"), # Smart single quote
        #('‘', "'"), # Smart single quote
        #('”', '"'), # Smart double quote
        #('“', '"'), # Smart double quote
        #('–', '-'), # En dash
        #('—', '-'), # Em dash
        #('−', '-'), # Minus sign
        
        # --- Ellipsis ---
        #('…', '...'), # Unicode ellipsis

        # --- Invisible/Control Characters (remove them) ---
        ('\u200b', ''), # Zero-width space
        ('\u00ad', ''), # Soft hyphen
        ('\ufeff', ''), # Byte Order Mark (BOM)

        # --- Whitespace Variants (normalize to standard space) ---
        ('\u00a0', ' '), # Non-breaking space
        ('\u2009', ' '), # Thin space
        ('\u200a', ' '), # Hair space
        ('\u2002', ' '), # En space
        ('\u2003', ' '), # Em space
    ]
    for find, replace in normalization_rules:
        text = text.replace(find, replace)
    
    # IMPORTANT: Preserve newlines for the segmenter.
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

def _normalize_sentence_spacing_and_ending(sentence: str) -> str:
    """
    Normalizes whitespace and ensures sentence-ending punctuation for a single sentence.

    This function is applied after sentence segmentation.
    - It collapses all internal whitespace (including newlines, tabs, etc.) 
      into a single space.
    - It adds a period to the end of a sentence if it doesn't already end
      with a standard sentence-ending punctuation mark. It correctly
      handles sentences that end with quotes.
    """
    normalized_sentence = re.sub(r'\s+', ' ', sentence).strip()

    if not normalized_sentence:
        return ""

    # Add a period if the sentence doesn't end with punctuation.
    # Define sentence-ending punctuation and characters to ignore when checking.
    ending_punctuation = ".?!…"
    # Characters to strip from the end before checking (e.g., quotes).
    chars_to_strip_before_check = ' "\'’”' 
    
    trimmed_end = normalized_sentence.rstrip(chars_to_strip_before_check)
    
    # If the trimmed sentence is empty (e.g., input was just " ' "), do nothing.
    # Otherwise, check the last character of the actual content.
    if trimmed_end and trimmed_end[-1] not in ending_punctuation:
        normalized_sentence += '.'
        
    return normalized_sentence

def _prepare_and_segment_sentences(text: str, options: TextProcessingOptions) -> List[str]:
    """
    Executes the main text preparation pipeline to produce a clean list of
    synthesis-ready units (sentences or sub-sentences).

    Crucially, this function guarantees that no string in the output list will be
    longer than `options.max_chunk_length`.
    """
    # --- Step 1: Unicode and Whitespace Normalization ---
    text = _normalize_whitespace_and_unicode(text)
    if not text:
        return []

    # --- Step 2: Bracket Removal (Optional) ---
    if options.remove_bracketed_text:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
    
    # --- Step 3: Sentence Segmentation (pysbd) ---
    # pysbd works best on cased, properly punctuated text.
    segmenter = get_segmenter(options.text_language)
    sentences = segmenter.segment(text)

    # --- Step 4: Text Normalization (NeMo) ---
    if options.use_nemo_normalizer:
        # NeMo converts numbers, currencies, dates, etc., to their spoken form.
        normalizer = get_normalizer(options.text_language)
        sentences = normalizer.normalize_list(sentences)
    
    # --- Step 5: Post-processing and Force Splitting ---
    final_sentence_units = []
    for sent in sentences:
        # 5a. Normalize internal spacing and ensure final punctuation.
        sent = _normalize_sentence_spacing_and_ending(sent)
        # 5b. Optional: Apply advanced, heuristic cleaning rules on the isolated sentence.
        if options.apply_advanced_cleaning:
            # Handle stuttering/hesitation (e.g., "W-what?", "b-but")
            sent = re.sub(r'\b([A-Za-z])-(\1[A-Za-z]*)', r'\2', sent, flags=re.IGNORECASE)
            # Handle emphasis markers by removing only the markers: * and _.
            sent = re.sub(r'([*_])(.+?)\1', r'\2', sent)
        # 5c. Optional: Lowercasing is done *after* segmentation, as case is a cue for pysbd.
        if options.to_lowercase:
            sent = sent.lower()
        # 5d. Final strip
        sent = sent.strip()

        if not sent:
            continue # Ensure we don't add empty sentences.

        # GUARANTEE: Ensure no sentence unit is too long.
        # If a single normalized sentence exceeds the max length, split it now.
        final_sentence_units.extend(_force_split_long_chunk(sent, options.max_chunk_length))
            
    return final_sentence_units

from typing import List, Sequence

def _force_split_long_chunk(
    chunk: str,
    max_length: int,
    split_delimiters: Sequence[str] = (".", "?", "!", "…", "—", "–", ";", ":", ",", " ")
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
    
    Raises:
        ValueError: If max_length is not a positive integer.
    """
    if not chunk:
        return []
    
    # Add input validation to prevent infinite loops and errors.
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer.")

    results = []
    current_pos = 0
    while current_pos < len(chunk):
        remaining_chunk = chunk[current_pos:]

        # If the rest of the chunk is short enough, we're done.
        # This check also handles the initial case where the whole chunk is short enough.
        if len(remaining_chunk) <= max_length:
            results.append(remaining_chunk)
            break

        split_pos = -1

        # The search window must not exceed max_length to guarantee the output chunk size.
        search_window = remaining_chunk[:max_length]

        # Find the best delimiter by priority
        for delimiter in split_delimiters:
            # Find the last occurrence of this delimiter in the window
            pos = search_window.rfind(delimiter)
            if pos != -1:
                # We found a delimiter. The split should happen *after* it.
                split_pos = pos + len(delimiter) # Use len(delimiter) for multi-char delimiters
                break  # Exit after finding the highest-priority delimiter

        # If no delimiter was found, perform a hard cut at max_length
        if split_pos == -1:
            split_pos = max_length
        
        # Extract the piece and add it to results
        # .strip() is still useful here for cleaning up the end of the extracted piece.
        piece_to_add = remaining_chunk[:split_pos].strip()
        if piece_to_add:
            results.append(piece_to_add)
        
        # Move the main cursor forward
        current_pos += split_pos

        # Add a loop to consume whitespace between chunks.
        # This ensures the next chunk doesn't start with leftover spaces.
        while current_pos < len(chunk) and chunk[current_pos].isspace():
            current_pos += 1

    return results

def _calculate_chunk_cost(length: int, options: TextProcessingOptions) -> float:
    """Calculates the 'badness' of a chunk of a given length."""
    # This combination is impossible if inputs are pre-validated.
    # Cost is infinite to prevent the DP algorithm from ever choosing it.
    if length > options.max_chunk_length: return float('inf')
    cost = ((length - options.ideal_chunk_length) / options.ideal_chunk_length) ** 2
    if length < options.ideal_chunk_length * 0.5:
        cost *= options.shortness_penalty_factor
    return cost

def _chunk_sentences_balanced(sentences: List[str], options: TextProcessingOptions) -> List[str]:
    """Chunks sentences using dynamic programming for globally optimal splits."""
    if not sentences: return []
    n = len(sentences)
    sentence_lens = [len(s) for s in sentences]
    min_costs, split_points = [0.0] + [float('inf')] * n, [0] * (n + 1)

    for i in range(1, n + 1):
        current_chunk_len = -1
        for j in range(i, 0, -1):
            current_chunk_len += sentence_lens[j-1] + 1
            if current_chunk_len > options.max_chunk_length: 
                break # This potential chunk is invalid
            
            cost = _calculate_chunk_cost(current_chunk_len, options)
            total_cost = min_costs[j-1] + cost
            if total_cost < min_costs[i]:
                min_costs[i], split_points[i] = total_cost, j-1
    
    chunks, end = [], n
    while end > 0:
        start = split_points[end]
        chunks.append(" ".join(sentences[start:end]))
        end = start
    
    final_chunks = chunks[::-1]
    # This check acts as an assertion. If any chunk is too long, it indicates a
    # flaw in the DP logic itself, and we should fail fast.
    for chunk in final_chunks:
        if len(chunk) > options.max_chunk_length:
            raise ValueError(
                f"Balanced chunking produced an oversized chunk ({len(chunk)} > "
                f"{options.max_chunk_length}). This indicates a logic error."
            )
    return final_chunks

def _chunk_sentences_greedy(sentences: List[str], options: TextProcessingOptions) -> List[str]:
    """Groups sentences greedily. Fast but can create uneven chunks."""
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
        elif len(current_chunk) + len(sentence) + 1 <= options.max_chunk_length:
            current_chunk += " " + sentence
        else:
            if len(current_chunk) > options.max_chunk_length:
                raise ValueError(
                    f"Greedy chunking produced an oversized chunk ({len(current_chunk)} > "
                    f"{options.max_chunk_length}). This indicates a logic error."
                )
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        if len(current_chunk) > options.max_chunk_length:
            raise ValueError(
                f"Greedy chunking produced an oversized chunk ({len(current_chunk)} > "
                f"{options.max_chunk_length}). This indicates a logic error."
            )
        chunks.append(current_chunk)
    return chunks

def process_and_chunk_text(text: str, options: TextProcessingOptions) -> List[str]:
    """
    The main entry point for cleaning, normalizing, and chunking text for TTS.
    """
    # The Paragraph strategy has a special case where it processes paragraph by paragraph.
    # For all other strategies, we prepare sentences from the whole text at once.
    
    strategy = options.chunking_strategy
    
    if strategy == TextChunkingStrategy.SIMPLE:
        # For 'SIMPLE', we prepare sentences to get all normalization benefits,
        # but then we join and re-split them without regard for sentence boundaries,
        # which is the spirit of this strategy.
        sentences = _prepare_and_segment_sentences(text, options)
        full_processed_text = " ".join(sentences)
        chunks = _force_split_long_chunk(full_processed_text, options.max_chunk_length)
    
    else: # PARAGRAPH, BALANCED, or GREEDY
        # These strategies all operate on pre-segmented, length-validated units.
        chunks = []
        if strategy == TextChunkingStrategy.PARAGRAPH:
            paragraphs = re.split(r'\n{2,}', text.strip())
            for para in paragraphs:
                if not para.strip(): continue
                sentences = _prepare_and_segment_sentences(para, options)
                chunks.extend(_chunk_sentences_balanced(sentences, options))
        elif strategy == TextChunkingStrategy.BALANCED:
            sentences = _prepare_and_segment_sentences(text, options)
            chunks = _chunk_sentences_balanced(sentences, options)
        elif strategy == TextChunkingStrategy.GREEDY:
            sentences = _prepare_and_segment_sentences(text, options)
            chunks = _chunk_sentences_greedy(sentences, options)
        else:
            raise ValueError(f"Unknown chunking strategy: '{strategy}'")
        
    final_chunks = [chunk for chunk in chunks if chunk]
    logging.info(f"Processed text into {len(final_chunks)} chunks using '{strategy}' strategy.")
    return final_chunks