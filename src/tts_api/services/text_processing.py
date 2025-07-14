import re
import logging
from typing import List, Sequence
from functools import lru_cache
from enum import Enum, auto
from dataclasses import dataclass
import unicodedata

import regex
import pysbd
from nemo_text_processing.text_normalization.normalize import Normalizer

from tts_api.core.config import settings
from tts_api.core.models import TextChunkingStrategy, TextProcessingOptions

PARAGRAPH_BREAK_TOKEN = "__PARAGRAPH_BREAK__"

# For _clean_unicode_characters
_UNICODE_WHITELIST_PATTERN = regex.compile(r'[^\p{L}\p{N}\p{P}\p{S}\s]+')
_UNICODE_BLACKLIST_PATTERN = regex.compile(
    r'[\p{Emoji_Presentation}\p{InBox_Drawing}\p{InBlock_Elements}\p{InGeometric_Shapes}\p{InMiscellaneous_Symbols}\p{InMiscellaneous_Symbols_and_Pictographs}]+'
)

class BoundaryType(Enum):
    """Describes the type of boundary that follows a text chunk."""
    SENTENCE = auto()      # A standard sentence break.
    PARAGRAPH = auto()     # A paragraph break (e.g., from a double newline).
    HARD_LIMIT = auto()    # A break forced by character limit, not semantics.

@dataclass
class TextChunk:
    """A container for a text chunk and its associated boundary metadata."""
    text: str
    # The type of break that FOLLOWS this chunk.
    boundary_type: BoundaryType

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

def _normalize_unicode(text: str) -> str:
    """
    Performs Unicode character normalization.

    Stage 1: Manually replace a curated list of characters that NFKC normalization
             is confirmed *not* to handle, or for which a specific TTS-friendly
             policy is desired (e.g., converting Japanese quotes to standard quotes).
    Stage 2: Use NFKC normalization to handle a wide range of other Unicode
             compatibility characters automatically, including various space types,
             ligatures, full-width characters, and mathematical symbols.
    """
    # Stage 1: Explicit manual mapping for policy-driven changes.
    manual_replacement_map = {
        # --- Quotes ---
        #'’': "'",  # Smart Single Quote (right)
        #'‘': "'",  # Smart Single Quote (left)
        #'”': '"',  # Smart Double Quote (right)
        #'“': '"',  # Smart Double Quote (left)

        # --- Dashes ---
        #'–': '-',  # En Dash
        #'—': '-',  # Em Dash
        #'−': '-',  # Minus Sign

        # --- Language-Specific Brackets (Policy: Convert to smart double quotes) ---
        '「': '“', '」': '”',  # Corner brackets to quotes
        '『': '“', '』': '”',  # Lenticular brackets to quotes
    }
    for find, replace in manual_replacement_map.items():
        text = text.replace(find, replace)

    # Stage 2: General Unicode normalization.
    text = unicodedata.normalize('NFKC', text)
    return text

def _clean_unicode_characters(text: str) -> str:
    """
    Removes non-verbalizable characters using a two-step approach.

    1.  Whitelist Pass: Keeps only characters from essential Unicode categories
        (\p{L}, \p{N}, \p{P}, \p{S}, \s), removing everything else (like control
        characters \p{C} and combining marks \p{M}).

    2.  Blacklist Pass: Removes specific categories of symbols that were
        allowed by the whitelist but are generally non-verbalizable. This
        includes Emojis, box-drawing characters, and various pictographs.
    """
    # Step 1: Whitelist. Remove anything NOT in the allowed categories.
    # This is a broad filter that cleans up control characters, format characters,
    # private-use characters, and unassigned code points.
    text = _UNICODE_WHITELIST_PATTERN.sub(' ', text)

    # Step 2: Blacklist. Remove specific unwanted categories that the
    # whitelist allowed through (because they are in \p{S}).
    text = _UNICODE_BLACKLIST_PATTERN.sub(' ', text)
    
    return text

def _normalize_whitespace(text: str) -> str:
    """
    Normalizes all whitespace in the text.

    - Normalizes all line endings to a single newline ('\n').
    - Collapses sequences of horizontal whitespace (spaces, tabs) into one space.
    - Collapses multiple newlines, preserving paragraph breaks (double newlines).
    - Removes leading/trailing whitespace from the text.
    """
    # 3a: Normalize all line endings to LF ('\n').
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 3b: Collapse horizontal whitespace into a single space.
    text = re.sub(r'[ \t]+', ' ', text)

    # 3c: Collapse vertical whitespace blocks, preserving paragraph breaks.
    def collapse_newline_whitespace(match):
        if '\n\n' in match.group(0):
            return '\n\n'
        else:
            return '\n'
    text = re.sub(r'(\s*\n\s*)+', collapse_newline_whitespace, text)

    return text.strip()

def _normalize_repeated_punctuation(text: str) -> str:
    """
    Applies a set of ordered, best-practice rules to normalize repeated punctuation.
    The order is critical to handle complex cases before simple ones.
    """
    # Rule 1 (Highest Priority): Handle interleaved interrobangs.
    text = re.sub(r'(!\?|\?!){2,}', '?!', text)
    # Rule 2: Normalize sequences of hyphens to a proper em-dash for dialogue.
    text = re.sub(r'--+', '—', text)
    # Rule 3: Normalize sequences of periods to a standard ellipsis.
    text = re.sub(r'\.{4,}', '...', text)
    # Rule 4 (General Fallback): Collapse any other single repeating punctuation
    # mark (3 or more times) down to a single instance.
    text = re.sub(r'([!?,;:])\1{2,}', r'\1', text)
    return text

def _normalize_global_stylistic_patterns(text: str, options: TextProcessingOptions) -> str:
    """
    Applies normalizations that affect the global structure of the text,
    such as paragraph breaks. This must run before sentence segmentation.
    """
    
    # The order of these operations is critical for correctness.
    
    # 1. Normalize scene breaks (highest priority, line-based context).
    if options.normalize_scene_breaks:
        def scene_break_replacer(match: re.Match) -> str:
            symbols_only = re.sub(r'\s', '', match.group(0))
            if len(symbols_only) >= options.min_scene_break_length:
                return '\n\n'
            return match.group(0)
        text = re.sub(r'^[^\w\n]+$', scene_break_replacer, text, flags=re.MULTILINE)

    # 2. Normalize specific, defined punctuation patterns.
    # This runs BEFORE the general symbol rule to handle cases like '---' -> '—'.
    if options.normalize_repeated_punctuation:
        text = _normalize_repeated_punctuation(text)
    
    # 3. Truncate repeated alphanumeric characters.
    if options.max_repeated_alpha_chars is not None:
        limit = options.max_repeated_alpha_chars
        pattern = re.compile(r'([a-zA-Z])\1{' + str(limit) + r',}', re.IGNORECASE)
        text = pattern.sub(lambda m: m.group(1) * limit, text)

    # 4. General "catch-all" for any other repeated symbol.
    # This runs AFTER the specific punctuation rules.
    if options.max_repeated_symbol_chars is not None:
        limit = options.max_repeated_symbol_chars
        # This regex targets any non-alphanumeric, non-whitespace character that
        # isn't part of the specific punctuation rules handled above.
        # It's a robust fallback for things like '$$$', '###', '***'.
        pattern = re.compile(r'([^\w\s.?!—,:;])\1{' + str(limit) + r',}')
        text = pattern.sub(lambda m: m.group(1) * limit, text)

    return text

def _normalize_local_stylistic_patterns(text: str, options: TextProcessingOptions) -> str:
    """
    Applies normalizations that affect intra-sentence patterns. This can
    safely run on individual sentences after segmentation.
    """
    # 1. Normalize stuttering patterns.
    if options.normalize_stuttering:
        # Correctly handles case and repeated stutters (e.g., W-w-what -> What).
        text = re.sub(r'\b([A-Za-z])(?:[-.]\1)+', r'\1', text, flags=re.IGNORECASE)

    # 2. Normalize emphasis markers.
    if options.normalize_emphasis:
        # First, handle double markers (e.g., **word** or __word__) to ensure they match.
        text = re.sub(r'(\*\*|__)(.+?)\1', r'\2', text)
        # Then, handle single markers (e.g., *word* or _word_).
        text = re.sub(r'(\*|_)(.+?)\1', r'\2', text)
    
    return text

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
    ending_punctuation = ".?!"
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
    Executes the main text preparation pipeline using a multi-stage process
    to produce a clean list of synthesis-ready units.
    """
    # Step 1: Clean and normalize text.
    text = _normalize_unicode(text)
    text = _clean_unicode_characters(text)
    text = _normalize_whitespace(text)
    if not text:
        return []
    
    # Step 2: Apply global normalizations that affect text structure (e.g., scene breaks)
    # This must run BEFORE sentence segmentation.
    text = _normalize_global_stylistic_patterns(text, options)

    # Step 3: Bracket Removal (Optional)
    if options.remove_bracketed_text:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)

    # Step 4: Sentence Segmentation using the cleaned, structured text.
    segmenter = get_segmenter(options.text_language)
    sentences = segmenter.segment(text)

    # Step 5. Apply local, intra-sentence stylistic normalizations.
    sentences = [_normalize_local_stylistic_patterns(sent, options) for sent in sentences]

    # Step 6: Text Normalization (NeMo) for numbers, dates, etc.
    if options.use_nemo_normalizer:
        normalizer = get_normalizer(options.text_language)
        sentences = normalizer.normalize_list(sentences)
    
    # Step 7: Tokenize Paragraph Breaks.
    tokenized_sentences = []
    for sent in sentences:
        # If a sentence contains a paragraph break, split it and insert our token.
        if '\n\n' in sent:
            # Split the sentence by the paragraph break.
            parts = sent.split('\n\n')
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    tokenized_sentences.append(part)
                # Add the token between the parts, effectively replacing the '\n\n'.
                if i < len(parts) - 1:
                    tokenized_sentences.append(PARAGRAPH_BREAK_TOKEN)
        else:
            # If there's no break, add the sentence as is.
            tokenized_sentences.append(sent)

    # Step 8: Final per-sentence processing loop.
    final_sentence_units = []
    for sent in tokenized_sentences:
        # If the sentence is our special token, pass it through directly.
        if sent == PARAGRAPH_BREAK_TOKEN:
            final_sentence_units.append(PARAGRAPH_BREAK_TOKEN)
            continue
        # Otherwise, it's a sentence that needs final normalization.

        # 8a. Normalize internal spacing and ensure final punctuation.
        sent = _normalize_sentence_spacing_and_ending(sent)
        # 8b. Optional: Lowercasing.
        if options.to_lowercase:
            sent = sent.lower()
        # 8c. Final strip to remove any leading/trailing whitespace.
        sent = sent.strip()

        if not sent:
            continue # Ensure we don't add empty sentences.

        # GUARANTEE: Ensure no sentence unit is too long.
        # If a single normalized sentence exceeds the max length, split it now.
        final_sentence_units.extend(_force_split_long_chunk(sent, options.max_chunk_length))
            
    return final_sentence_units

def _force_split_long_chunk(
    chunk: str,
    max_length: int,
    split_delimiters: Sequence[str] = (".", "?", "!", "—", "–", ";", ":", ",", " ")
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

def process_and_chunk_text(text: str, options: TextProcessingOptions) -> List[TextChunk]:
    """
    The main entry point for cleaning, normalizing, and chunking text for TTS.
    This function consumes a tokenized stream of sentences and paragraph markers
    to produce the final list of TextChunk objects.
    """
    # Step 1: Get the clean, tokenized stream of sentences and paragraph markers.
    all_units = _prepare_and_segment_sentences(text, options)
    
    # Step 2: Reconstruct paragraph groupings based on the token.
    sentence_groups: List[List[str]] = []
    current_group: List[str] = []
    for unit in all_units:
        if unit == PARAGRAPH_BREAK_TOKEN:
            if current_group:
                sentence_groups.append(current_group)
            current_group = []
        else:
            current_group.append(unit)
    if current_group:
        sentence_groups.append(current_group)

    # Step 3: Apply the final chunking strategy to the sentence groups.
    strategy = options.chunking_strategy
    final_chunks: List[TextChunk] = []

    if strategy == TextChunkingStrategy.SIMPLE:
        # For SIMPLE, we ignore paragraph structure.
        full_processed_text = " ".join([s for group in sentence_groups for s in group])
        chunks_text = _force_split_long_chunk(full_processed_text, options.max_chunk_length)
        if chunks_text:
            # All but the last chunk are hard-limit splits.
            for chunk in chunks_text[:-1]:
                final_chunks.append(TextChunk(chunk, BoundaryType.HARD_LIMIT))
            final_chunks.append(TextChunk(chunks_text[-1], BoundaryType.SENTENCE))

    else: # PARAGRAPH, BALANCED, or GREEDY
        chunker_func = {
            TextChunkingStrategy.PARAGRAPH: _chunk_sentences_balanced,
            TextChunkingStrategy.BALANCED: _chunk_sentences_balanced,
            TextChunkingStrategy.GREEDY: _chunk_sentences_greedy,
        }.get(strategy)

        # For BALANCED/GREEDY, treat the whole text as a single logical group.
        if strategy != TextChunkingStrategy.PARAGRAPH:
            all_sentences_flat = [s for group in sentence_groups for s in group]
            sentence_groups = [all_sentences_flat] if all_sentences_flat else []

        for i, group in enumerate(sentence_groups):
            if not group: continue
            
            group_chunks_text = chunker_func(group, options)
            if not group_chunks_text: continue
            
            # Assign boundary types to the chunks of this group.
            for chunk in group_chunks_text[:-1]:
                final_chunks.append(TextChunk(chunk, BoundaryType.SENTENCE))
            
            # The last chunk of a group is special.
            is_last_group = (i == len(sentence_groups) - 1)
            # Use a PARAGRAPH break unless it's the very end of the text.
            boundary = (
                BoundaryType.PARAGRAPH if strategy == TextChunkingStrategy.PARAGRAPH and not is_last_group
                else BoundaryType.SENTENCE
            )
            final_chunks.append(TextChunk(group_chunks_text[-1], boundary))

    # Filter out any empty chunks that might have been created
    final_chunks = [chunk for chunk in final_chunks if chunk.text]

    logging.info(f"Processed text into {len(final_chunks)} chunks using '{strategy}' strategy.")
    return final_chunks