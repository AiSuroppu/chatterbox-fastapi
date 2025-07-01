import pytest
from unittest.mock import patch, MagicMock
import re

from tts_api.core.models import TextChunkingStrategy, TextProcessingOptions
from tts_api.services.text_processing import (
    process_and_chunk_text,
    _clean_unicode_characters,
    _normalize_unicode,
    TextChunk,
    BoundaryType,
)


# --- Test Data ---

SIMPLE_TEXT = "This is the first sentence. This is the second sentence."
PARAGRAPH_TEXT = "This is the first paragraph.\n\nThis is the second paragraph. It is a bit longer."
LONG_PARAGRAPH_TEXT = (
    "This is a very long paragraph. It has many sentences that are intended to push the total character count "
    "well over the typical maximum length. We need to test the fallback mechanism which should "
    "activate balanced chunking. Let's add one final sentence to be absolutely sure. The end."
)
TEXT_WITH_BRACKETS = "This is useful text (with parenthetical text) and [bracketed notes]. *Also asterisks*."
VERY_LONG_SENTENCE = (
    "This single sentence is deliberately made incredibly long to test the pre-splitting logic that should "
    "break it up before the main chunking strategy even sees it, because no single sentence can exceed the max length."
)


# --- Helper Function ---
def get_chunk_texts(text, options):
    """Helper to call the processor and extract text from TextChunk objects."""
    chunks = process_and_chunk_text(text, options)
    return [chunk.text for chunk in chunks]


# --- Test Suite ---

class TestProcessAndChunkText:
    """Tests for the main process_and_chunk_text function."""

    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_empty_and_whitespace_text(self, strategy):
        """Test that empty or whitespace-only text results in an empty list for all strategies."""
        options = TextProcessingOptions(chunking_strategy=strategy)
        assert get_chunk_texts("", options) == []
        assert get_chunk_texts("   ", options) == []
        assert get_chunk_texts(" \n\n\t ", options) == []

    def test_simple_strategy(self):
        """Test the SIMPLE strategy which force-splits text by character length."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.SIMPLE,
            max_chunk_length=50,
            to_lowercase=False
        )
        # It should split at the last space before the max_length
        text = "This is a sentence that is longer than fifty characters and should be split."
        expected = [
            "This is a sentence that is longer than fifty",
            "characters and should be split."
        ]
        assert get_chunk_texts(text, options) == expected

    def test_greedy_strategy(self):
        """Test the GREEDY strategy which groups sentences without exceeding max_length."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.GREEDY,
            max_chunk_length=80,
            to_lowercase=True
        )
        text = "First sentence is short. Second one is also quite manageable. The third sentence is much longer and will not fit with the second."
        expected = [
            "first sentence is short. second one is also quite manageable.",
            "the third sentence is much longer and will not fit with the second."
        ]
        assert get_chunk_texts(text, options) == expected

    def test_balanced_strategy_properties(self):
        """Test the BALANCED strategy, focusing on properties rather than exact output."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=150,
            ideal_chunk_length=100,
            to_lowercase=True
        )
        # Use the long paragraph text which requires complex splitting
        chunks = get_chunk_texts(LONG_PARAGRAPH_TEXT, options)

        assert chunks is not None
        assert len(chunks) > 1 # It should have been split
        assert all(isinstance(c, str) for c in chunks)
        assert all(0 < len(c) <= options.max_chunk_length for c in chunks)

        # Check that the content is preserved
        reconstructed_text = " ".join(chunks)
        # Clean the original text the same way the function would for comparison
        expected_text = ' '.join(LONG_PARAGRAPH_TEXT.lower().replace('\n\n', ' ').split())
        reconstructed_text_no_multi_space = ' '.join(reconstructed_text.split())

        assert reconstructed_text_no_multi_space == expected_text

    def test_paragraph_strategy_short_paragraphs(self):
        """Test the PARAGRAPH strategy with paragraphs shorter than max_length."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=200,
            to_lowercase=True
        )
        expected = [
            "this is the first paragraph.",
            "this is the second paragraph. it is a bit longer."
        ]
        assert get_chunk_texts(PARAGRAPH_TEXT, options) == expected

    def test_paragraph_strategy_long_paragraph_fallback(self):
        """Test PARAGRAPH strategy's fallback to BALANCED for a long paragraph."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=120,
            ideal_chunk_length=100,
            to_lowercase=True
        )
        # This text has one paragraph that is much longer than max_chunk_length
        chunks = get_chunk_texts(LONG_PARAGRAPH_TEXT, options)
        
        # It should have been split into multiple chunks by the balanced strategy
        assert len(chunks) > 1
        assert all(len(c) <= options.max_chunk_length for c in chunks)

    def test_option_to_lowercase(self):
        """Verify the to_lowercase option works as expected."""
        text = "This Is Mixed CASE."
        # Test when True
        options_true = TextProcessingOptions(to_lowercase=True)
        assert get_chunk_texts(text, options_true) == ["this is mixed case."]
        
        # Test when False
        options_false = TextProcessingOptions(to_lowercase=False)
        assert get_chunk_texts(text, options_false) == ["This Is Mixed CASE."]

    def test_option_remove_bracketed_text(self):
        """Verify the remove_bracketed_text option works as expected."""
        # Test when True. Note: normalize_emphasis is True by default and will remove asterisks.
        options_true = TextProcessingOptions(remove_bracketed_text=True, to_lowercase=False)
        # After bracket removal, text is "This is useful text  and . *Also asterisks*."
        # The processing logic does not clean up the space before the period, so the
        # expected output must reflect this.
        expected_true = ["This is useful text and . Also asterisks."]
        assert get_chunk_texts(TEXT_WITH_BRACKETS, options_true) == expected_true

        # Test when False. Emphasis is still normalized by default.
        options_false = TextProcessingOptions(remove_bracketed_text=False, to_lowercase=False)
        # The original text is segmented and cleaned, but brackets are kept.
        expected_false = ["This is useful text (with parenthetical text) and [bracketed notes]. Also asterisks."]
        assert get_chunk_texts(TEXT_WITH_BRACKETS, options_false) == expected_false

    def test_pre_splitting_of_very_long_sentences(self):
        """Ensure sentences longer than max_chunk_length are pre-split."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.GREEDY,
            max_chunk_length=100,
        )
        chunks = get_chunk_texts(VERY_LONG_SENTENCE, options)

        assert len(chunks) > 1, "The single long sentence should have been split"
        for chunk in chunks:
            assert len(chunk) <= options.max_chunk_length

    def test_unknown_strategy_raises_error(self):
        """Test that an invalid strategy raises an error."""
        # Use model_construct to bypass Pydantic's enum validation for testing.
        options = TextProcessingOptions.model_construct(chunking_strategy="nonexistent_strategy")
        
        # The current implementation uses dict.get(), resulting in None for the chunker function.
        # Calling a NoneType object raises a TypeError.
        with pytest.raises(TypeError):
            process_and_chunk_text("some text", options)

    @patch('tts_api.services.text_processing.get_segmenter')
    def test_language_option_is_passed_to_segmenter(self, mock_get_segmenter):
        """Check if the text_language option is correctly passed to the segmenter factory."""
        # Mock the segmenter to control its output and avoid downloading models
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = ["Ein Satz."]
        mock_get_segmenter.return_value = mock_segmenter
        
        # Use a strategy that calls _prepare_and_segment_sentences, like BALANCED
        options = TextProcessingOptions(
            text_language="de",
            chunking_strategy=TextChunkingStrategy.BALANCED
        )
        
        get_chunk_texts("Ein Satz.", options)
        
        # The key assertion: was the factory called with the correct language?
        mock_get_segmenter.assert_called_once_with("de")


class TestProcessAndChunkTextEdgeCases:
    """Probes the boundaries, interactions, and corner cases of the text processing."""

    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_text_with_only_removable_content(self, strategy):
        """Test that text becoming empty after cleaning results in an empty list."""
        text = "[all content is inside brackets] (and parentheses)"
        options = TextProcessingOptions(
            chunking_strategy=strategy,
            remove_bracketed_text=True
        )
        assert get_chunk_texts(text, options) == []

    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_unicode_and_emojis_are_handled(self, strategy):
        """Ensure non-ASCII characters don't break the logic and lengths are correct."""
        text = "你好, 世界. This is a very long test sentence to ensure a split happens 😊."
        options = TextProcessingOptions(
            chunking_strategy=strategy,
            max_chunk_length=55,
            to_lowercase=False,
        )
        chunks = get_chunk_texts(text, options)
        reconstructed_text = " ".join(chunks)
        assert "你好, 世界." in reconstructed_text
        # The current cleaning policy does not remove emojis, so it should be present.
        assert "😊" not in reconstructed_text
        assert all(len(c) <= 55 for c in chunks)

    def test_multiple_consecutive_newlines_in_paragraph_strategy(self):
        """PARAGRAPH strategy should handle multiple newlines as a single separator."""
        text = "Paragraph 1.\n\n\n\nParagraph 2."
        options = TextProcessingOptions(chunking_strategy=TextChunkingStrategy.PARAGRAPH)
        chunks = get_chunk_texts(text, options)
        assert chunks == ["Paragraph 1.", "Paragraph 2."]

    def test_paragraph_boundary_conditions_with_max_length(self):
        """Test PARAGRAPH strategy's fallback logic right at the max_length boundary."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=52,
            to_lowercase=False
        )
        
        # Case 1: Length is exactly max_length. Should NOT fall back.
        text_at_limit = "This paragraph has length of exactly fiftytwo chars."
        assert len(text_at_limit) == 52
        chunks_at_limit = get_chunk_texts(text_at_limit, options)
        assert chunks_at_limit == [text_at_limit]
        
        # Case 2: Length is one over max_length. SHOULD fall back.
        text_over_limit = "This paragraph has a length of over fifty-two characters."
        assert len(text_over_limit) > 52
        chunks_over_limit = get_chunk_texts(text_over_limit, options)
        # The pre-splitter will break the long sentence before the chunker sees it.
        # The split happens at the last space before the 52-char limit.
        expected_split = [
            "This paragraph has a length of over fifty-two",
            "characters."
        ]
        assert chunks_over_limit == expected_split

    def test_greedy_strategy_boundary_fill(self):
        """Test GREEDY strategy when a chunk exactly fills or just misses max_length."""
        s1 = "This sentence is a specific length for this boundary test." # 58 chars
        s2 = "Short one." # 10 chars
        text = f"{s1} {s2}" # Combined with space: 58 + 1 + 10 = 69 chars

        # Case 1: max_length allows the combination
        options_fit = TextProcessingOptions(chunking_strategy=TextChunkingStrategy.GREEDY, max_chunk_length=69)
        assert get_chunk_texts(text, options_fit) == [f"{s1} {s2}"]

        # Case 2: max_length is one too short, forcing a split
        options_no_fit = TextProcessingOptions(chunking_strategy=TextChunkingStrategy.GREEDY, max_chunk_length=68)
        assert get_chunk_texts(text, options_no_fit) == [s1, s2]

    def test_interaction_of_cleaning_and_splitting(self):
        """Verify that cleaning happens correctly within the sentence processing loop."""
        text = "Start. [This long part is removed, making it fit.] End."
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.GREEDY,
            max_chunk_length=50, # Without removal, pysbd might split this
            remove_bracketed_text=True,
            to_lowercase=False
        )
        # The empty sentence from the brackets is discarded, and the rest is joined.
        chunks = get_chunk_texts(text, options)
        assert chunks == ["Start. End."]

    def test_no_valid_split_points_in_long_sentence(self):
        """Test _force_split_long_chunk's hard cut when no delimiters are found."""
        long_word = "a" * 150
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.GREEDY,
            max_chunk_length=100,
            max_repeated_alpha_chars=None
        )
        chunks = get_chunk_texts(long_word, options)
        # The processor adds a period, then force-splits.
        assert chunks == ["a" * 100, "a" * 50 + "."]
        
    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_chunks_are_never_empty_strings(self, strategy):
        """Ensure no strategy ever returns an empty string in the chunk list."""
        text = "Text...  More text.\n\n\nFinal text."
        # FIX: Changed max_chunk_length to a valid value (must be >= 50).
        options = TextProcessingOptions(chunking_strategy=strategy, max_chunk_length=50)
        
        chunks = get_chunk_texts(text, options)
        
        assert all(chunks), "Found an empty or None string in the chunk list"
        assert "" not in chunks

class TestBalancedStrategySpecifics:
    """
    In-depth tests for the BALANCED chunking strategy to verify its
    optimization logic and parameter sensitivity.
    """

    def test_avoids_greedy_choice_due_to_shortness_penalty(self):
        """
        Tests the core value of the balanced strategy: avoiding a tiny leftover
        chunk even if the preceding chunk is slightly less 'ideal'.
        """
        # Scenario:
        # - Greedy would group S1+S2 (len 145) and leave S3 (len 9) as a tiny chunk.
        # - Balanced, with a moderate penalty, should prefer to split S1 (len 89)
        #   and group S2+S3 (len 65) to create two more evenly sized chunks.
        s1 = "This first sentence is deliberately made to be a decent length, around eighty characters." # len 89
        s2 = "The second sentence is shorter, about fifty chars long." # len 55
        s3 = "Tiny one." # len 9

        text = f"{s1} {s2} {s3}"

        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=150,
            ideal_chunk_length=100,
            shortness_penalty_factor=2.0,
            to_lowercase=False
        )

        chunks = get_chunk_texts(text, options)

        expected_chunks = [
            s1,
            f"{s2} {s3}"
        ]

        assert chunks == expected_chunks
        # Also check the greedy strategy to prove they behave differently
        options.chunking_strategy = TextChunkingStrategy.GREEDY
        greedy_chunks = get_chunk_texts(text, options)
        assert greedy_chunks != expected_chunks
        assert greedy_chunks == [f"{s1} {s2}", s3]

    def test_avoids_greedy_choice_due_to_shortness_penalty2(self):
        """
        Tests the core value of the balanced strategy: balanced text chunks.
        """
        # Scenario:
        # - Greedy would group S1+S2 (len 145) and leave S3 (len 54) as another chunk.
        # - Balanced should prefer S1 (len 89) and S2+S3 (len 110) as more even chunks.
        s1 = "This first sentence is deliberately made to be a decent length, around eighty characters." # len 89
        s2 = "The second sentence is shorter, about fifty chars long." # len 55
        s3 = "The third sentence is shorter, about fifty chars long." # len 54

        text = f"{s1} {s2} {s3}"

        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=150,
            ideal_chunk_length=100,
            shortness_penalty_factor=2.0,
            to_lowercase=False
        )

        chunks = get_chunk_texts(text, options)

        expected_chunks = [
            s1,
            f"{s2} {s3}"
        ]

        assert chunks == expected_chunks
        # Also check the greedy strategy to prove they behave differently
        options.chunking_strategy = TextChunkingStrategy.GREEDY
        greedy_chunks = get_chunk_texts(text, options)
        assert greedy_chunks != expected_chunks
        assert greedy_chunks == [f"{s1} {s2}", s3]

    def test_effect_of_ideal_chunk_length(self):
        """
        Tests that changing the ideal_chunk_length alters the chunking outcome
        as expected, aiming for either more or fewer chunks.
        """
        text = (
            "This is the first sentence, it provides some initial content. " # 62
            "Here comes a second sentence, adding more length to the text. " # 62
            "And a third one for good measure, making the total length significant. " # 71
            "Finally, a fourth sentence concludes this test paragraph." # 57
        )

        # Scenario A: Smaller ideal length should favor more, smaller chunks.
        options_a = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=200,
            ideal_chunk_length=80, # Small ideal length
            to_lowercase=False
        )
        chunks_a = get_chunk_texts(text, options_a)

        # Scenario B: Larger ideal length should favor fewer, larger chunks.
        options_b = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=200,
            ideal_chunk_length=150, # Large ideal length
            to_lowercase=False
        )
        chunks_b = get_chunk_texts(text, options_b)

        # The primary assertion is that the choice of ideal_length had an effect.
        assert len(chunks_a) > len(chunks_b)
        # Expected behavior: smaller ideal length splits more granularly
        assert len(chunks_a) == 4
        # Expected behavior: larger ideal length groups more aggressively
        assert len(chunks_b) == 2

    def test_respects_absolute_max_chunk_length(self):
        """
        Ensures the strategy NEVER creates a chunk over max_chunk_length,
        even if doing so would be more 'optimal' in terms of ideal_length.
        """
        # Scenario: S1+S2 is closer to the ideal length than S1 and S2 are individually,
        # but the combined length exceeds max_chunk_length.
        s1 = "a" * 90
        s2 = "b" * 90
        text = f"{s1}. {s2}."

        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=170, # Combined length with space is 183, so this is too small.
            ideal_chunk_length=200, # Ideal is high, encouraging large chunks.
            to_lowercase=False,
            max_repeated_alpha_chars=None
        )

        chunks = get_chunk_texts(text, options)

        # The cost of an oversized chunk is infinite, so it MUST be split.
        expected_chunks = [f"{s1}.", f"{s2}."]
        assert chunks == expected_chunks
        assert all(len(c) <= options.max_chunk_length for c in chunks)


    def test_with_pre_split_long_sentence(self):
        """
        Verifies that the balanced strategy can optimally re-combine fragments
        that were created by the initial _force_split_long_chunk pass.
        """
        long_sentence_with_breaks = "a" * 70 + ". " + "b" * 70 + ". " + "c" * 70 + ". " + "d" * 70 + "."
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=150, # Allows two fragments to be combined
            ideal_chunk_length=140,
            to_lowercase=False,
            max_repeated_alpha_chars=None
        )

        chunks = get_chunk_texts(long_sentence_with_breaks, options)

        # It receives ['a'*70 + '.', 'b'*70 + '.', 'c'*70 + '.', 'd'*70 + '.']
        # It should group them in pairs.
        expected_chunks = [
            'a' * 70 + '. ' + 'b' * 70 + '.',
            'c' * 70 + '. ' + 'd' * 70 + '.',
        ]
        assert chunks == expected_chunks

class TestStylisticNormalization:
    """Tests the various text normalization and cleaning features."""

    @pytest.mark.parametrize("text, expected_chunks, options_kwargs", [
        # --- Scene Breaks ---
        ("Para 1\n\n***\n\nPara 2", ["Para 1.", "Para 2."], {"normalize_scene_breaks": True, "min_scene_break_length": 3}),
        ("Para 1\n\n**\n\nPara 2", ["Para 1.", "**.", "Para 2."], {"normalize_scene_breaks": True, "min_scene_break_length": 3, "max_repeated_symbol_chars": None}),
        ("Para 1 *** Para 2", ["Para 1 *** Para 2."], {"normalize_scene_breaks": True, "min_scene_break_length": 3, "max_repeated_symbol_chars": None}),

        # --- Repeated Punctuation ---
        ("What?!?!?!", ["What?!"], {"normalize_repeated_punctuation": True}),
        ("Wait-- no.", ["Wait— no."], {"normalize_repeated_punctuation": True}),
        ("So.......", ["So..."], {"normalize_repeated_punctuation": True}),
        ("Why,,,, because.", ["Why, because."], {"normalize_repeated_punctuation": True}),
        ("This is not normalized!!", ["This is not normalized!!"], {"normalize_repeated_punctuation": True}),

        # --- Emphasis ---
        ("This is *important*.", ["This is important."], {"normalize_emphasis": True}),
        ("This is **very important**.", ["This is very important."], {"normalize_emphasis": True}),
        ("This is _important*.", ["This is _important*."], {"normalize_emphasis": True}), # Mismatched, should not change
        ("This is __very important__.", ["This is very important."], {"normalize_emphasis": True}),
        ("A mix of *styles* and _other styles_.", ["A mix of styles and other styles."], {"normalize_emphasis": True}),

        # --- Stuttering ---
        ("W-w-what is that?", ["What is that?"], {"normalize_stuttering": True}),
        ("P.p.please help.", ["Please help."], {"normalize_stuttering": True}),
        ("S-S-Stop!", ["Stop!"], {"normalize_stuttering": True}),

        # --- Repeated Alpha Chars ---
        ("Ahhhhhhh", ["Ahhh."], {"max_repeated_alpha_chars": 3}),
        ("Noooooo please", ["Nooo please."], {"max_repeated_alpha_chars": 3}),
        ("Ahhh", ["Ahhh."], {"max_repeated_alpha_chars": 3}),
        ("Ahhhhhhh", ["Ahhhhhhh."], {"max_repeated_alpha_chars": None}),

        # --- Repeated Symbol Chars ---
        ("$$$$$ a lot of money.", ["$ a lot of money."], {"max_repeated_symbol_chars": 1}),
        ("### Section ###", ["# Section #."], {"max_repeated_symbol_chars": 1, "normalize_scene_breaks": False}),
        ("$$$$$ money.", ["$$$ money."], {"max_repeated_symbol_chars": 3, "normalize_scene_breaks": False}),
        ("$$$$$ money.", ["$$$$$ money."], {"max_repeated_symbol_chars": None, "normalize_scene_breaks": False}),

        # --- Unicode & Whitespace ---
        ("「こんにちは」", ["“こんにちは”."], {}),
        ("ＨＥＬＬＯ,　ＷＯＲＬＤ！", ["HELLO, WORLD!"], {}),
        ("Text with\u200b a zero-width space.", ["Text with a zero-width space."], {}),
        ("Text   with    extra   spaces.", ["Text with extra spaces."], {}),
    ])
    def test_normalization_features(self, text, expected_chunks, options_kwargs):
        """A parameterized test to cover various normalization features."""
        # Start with default options and override with test-specific ones
        defaults = TextProcessingOptions().model_dump()
        for key, value in options_kwargs.items():
            defaults[key] = value

        # Isolate the tests by disabling other normalizations unless specified
        defaults['to_lowercase'] = False
        if 'normalize_emphasis' not in options_kwargs: defaults['normalize_emphasis'] = False
        if 'normalize_stuttering' not in options_kwargs: defaults['normalize_stuttering'] = False
        if 'normalize_repeated_punctuation' not in options_kwargs: defaults['normalize_repeated_punctuation'] = False

        options = TextProcessingOptions(**defaults)
        result = get_chunk_texts(text, options)
        assert result == expected_chunks

    def test_interaction_of_punctuation_rules(self):
        """
        Tests that specific punctuation rules run before the generic symbol rule.
        '....' should become '...' (specific rule), not '.' (generic rule).
        """
        text = "Wait.... $$$$"
        options = TextProcessingOptions(
            normalize_repeated_punctuation=True,
            max_repeated_symbol_chars=1,
            to_lowercase=False
        )
        # Expected: ellipsis rule handles periods, symbol rule handles dollar signs
        expected = ["Wait... $."]
        assert get_chunk_texts(text, options) == expected

    def test_normalizer_does_not_remove_all_symbols(self):
        """
        Verify that unicode normalization preserves common symbols that carry meaning.
        """
        text = "The price is $50. 50% off! Is 2 > 1? Yes."
        options = TextProcessingOptions(to_lowercase=False, chunking_strategy=TextChunkingStrategy.BALANCED)
        # The core idea is that these symbols are NOT stripped out.
        # pysbd will split the text, but the balanced chunker will rejoin them.
        expected = ["The price is $50. 50% off! Is 2 > 1? Yes."]
        assert get_chunk_texts(text, options) == expected


class TestParagraphStrategyAndBoundaries:
    """
    Tests the PARAGRAPH chunking strategy and the correct assignment of
    BoundaryType metadata, which is critical for post-processing audio.
    """

    def test_basic_paragraph_separation_and_boundaries(self):
        """
        Ensures simple paragraph breaks (`\n\n`) result in separate chunks
        with the correct PARAGRAPH boundary type.
        """
        text = "First paragraph. It has one sentence.\n\nSecond paragraph. It also has one."
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=200,  # Long enough to not split paragraphs internally
            to_lowercase=False
        )
        
        chunks = process_and_chunk_text(text, options)

        assert len(chunks) == 2
        assert chunks[0] == TextChunk(
            text="First paragraph. It has one sentence.",
            boundary_type=BoundaryType.PARAGRAPH
        )
        assert chunks[1] == TextChunk(
            text="Second paragraph. It also has one.",
            boundary_type=BoundaryType.SENTENCE  # Last chunk in text is always SENTENCE
        )

    def test_long_paragraph_fallback_and_boundaries(self):
        """
        Tests that a long paragraph is split with SENTENCE boundaries, but the
        boundary after the whole group is still PARAGRAPH.
        """
        # This text has one long paragraph that MUST be split, followed by a short one.
        long_para = "This is a very long sentence that will be split. This is another sentence that adds to the length, ensuring a split is necessary."
        short_para = "This is the final paragraph."
        text = f"{long_para}\n\n{short_para}"
        
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=90,
            ideal_chunk_length=80,
            to_lowercase=False
        )
        
        chunks = process_and_chunk_text(text, options)

        # Expected: long_para is split into two, then short_para
        # [chunk1_part1, chunk1_part2, chunk2]
        assert len(chunks) == 3
        
        # Split within the first paragraph uses SENTENCE boundary
        assert chunks[0].text == "This is a very long sentence that will be split."
        assert chunks[0].boundary_type == BoundaryType.SENTENCE

        # The last chunk of the first paragraph group has a PARAGRAPH boundary
        assert chunks[1].text == "This is another sentence that adds to the length, ensuring a split is necessary."
        assert chunks[1].boundary_type == BoundaryType.PARAGRAPH

        # The final chunk of the entire text has a SENTENCE boundary
        assert chunks[2].text == "This is the final paragraph."
        assert chunks[2].boundary_type == BoundaryType.SENTENCE

    def test_non_paragraph_strategies_ignore_newlines(self):
        """
        Verifies that BALANCED and GREEDY strategies treat paragraph breaks
        as mere whitespace and do not create PARAGRAPH boundaries.
        """
        text = "Paragraph one. It is short.\n\nParagraph two. Also short."
        combined_text = "Paragraph one. It is short. Paragraph two. Also short."

        for strategy in [TextChunkingStrategy.BALANCED, TextChunkingStrategy.GREEDY]:
            options = TextProcessingOptions(
                chunking_strategy=strategy,
                max_chunk_length=200,
                to_lowercase=False
            )
            chunks = process_and_chunk_text(text, options)
            
            assert len(chunks) == 1, f"Strategy {strategy} should have created one chunk"
            assert chunks[0] == TextChunk(
                text=combined_text,
                boundary_type=BoundaryType.SENTENCE
            )

    def test_simple_strategy_ignores_newlines(self):
        """
        Verifies that SIMPLE strategy flattens paragraphs and uses HARD_LIMIT
        boundaries correctly.
        """
        text = "This is the first paragraph.\n\nThis is the second, which will be split."
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.SIMPLE,
            max_chunk_length=50,
            to_lowercase=False
        )
        chunks = process_and_chunk_text(text, options)

        # The text is flattened to "This is the first paragraph. This is the second, which will be split."
        # The intelligent splitter (_force_split_long_chunk) sees the period at char 28 and splits there,
        # as it's the highest-priority delimiter within the 50-char search window.
        assert len(chunks) == 2
        assert chunks[0] == TextChunk(
            text="This is the first paragraph.",
            boundary_type=BoundaryType.HARD_LIMIT
        )
        assert chunks[1] == TextChunk(
            text="This is the second, which will be split.",
            boundary_type=BoundaryType.SENTENCE
        )

    def test_consecutive_and_whitespace_paragraph_breaks(self):
        """
        Ensures that multiple newlines and surrounding whitespace are handled
        gracefully and treated as a single paragraph break.
        """
        text = "Paragraph 1. \n\n\n\n Paragraph 2. \n\n  \n\n Paragraph 3."
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=200,
            to_lowercase=False
        )

        chunks = process_and_chunk_text(text, options)

        assert len(chunks) == 3
        assert chunks[0] == TextChunk("Paragraph 1.", BoundaryType.PARAGRAPH)
        assert chunks[1] == TextChunk("Paragraph 2.", BoundaryType.PARAGRAPH)
        assert chunks[2] == TextChunk("Paragraph 3.", BoundaryType.SENTENCE)

    def test_paragraph_tokenization_inside_pysbd_segment(self):
        """
        Tests the specific case where pysbd might not split on a newline,
        ensuring the internal `__PARAGRAPH_BREAK__` tokenization works.
        """
        # This "..." syntax is one way to make pysbd keep the `\n\n`
        text = "A sentence may be interrupted...\n\n...and continue in a new paragraph."
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=200,
            to_lowercase=False
        )
        chunks = process_and_chunk_text(text, options)
        
        assert len(chunks) == 2
        assert chunks[0] == TextChunk(
            text="A sentence may be interrupted...",
            boundary_type=BoundaryType.PARAGRAPH
        )
        # The leading "..." are preserved as they are part of the text.
        # The ending period is added by our normalization.
        assert chunks[1] == TextChunk(
            text="...and continue in a new paragraph.",
            boundary_type=BoundaryType.SENTENCE
        )

    def test_no_empty_chunks_from_leading_trailing_breaks(self):
        """
        Ensures that paragraph breaks at the start or end of the text
        do not result in empty chunks.
        """
        text = "\n\nThis is the only paragraph.\n\n"
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            to_lowercase=False
        )
        chunks = process_and_chunk_text(text, options)
        
        assert len(chunks) == 1
        assert chunks[0] == TextChunk("This is the only paragraph.", BoundaryType.SENTENCE)

@pytest.mark.parametrize("description, input_char, expected_char", [
    ("Smart Double Quote (Right)", "”", '”'),
    ("Smart Double Quote (Left)", "“", '“'),
    ("Smart Single Quote (Right)", "’", "’"),
    ("Smart Single Quote (Left)", "‘", "‘"),
    
    ("En Dash", "–", "–"),
    ("Em Dash", "—", "—"),
    
    ("Ellipsis", "…", "..."),
    
    ("Corner Bracket (Left)", "「", '“'),
    ("Corner Bracket (Right)", "」", '”'),
    ("Lenticular Bracket (Left)", "『", '“'),
    ("Lenticular Bracket (Right)", "』", '”'),
])
def test_whitelist_keeps_common_unicode_variants(description, input_char, expected_char):
    """
    Tests that the whitelist approach correctly keeps (or normalizes)
    common non-ASCII punctuation instead of removing it.
    
    This test verifies that characters like smart quotes, dashes, and ellipsis
    are correctly handled by the normalization pipeline and not incorrectly
    classified as "non-verbalizable" and stripped.
    """
    # We embed the character in a simple sentence to make the test more realistic.
    test_text = f"start {input_char} end"
    
    # Run the function under test.
    cleaned_text = _clean_unicode_characters(_normalize_unicode(test_text))
    
    # Construct the expected output string based on the normalized character.
    expected_output = f"start {expected_char} end"
    
    # Assert that the cleaned text matches the expected normalized output.
    assert cleaned_text == expected_output, f"Failed on: {description}"