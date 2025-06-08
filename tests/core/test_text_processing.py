import pytest
from unittest.mock import patch, MagicMock

from tts_api.core.models import TextChunkingStrategy, TextProcessingOptions
from tts_api.utils.text_processing import process_and_chunk_text

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


# --- Test Suite ---

class TestProcessAndChunkText:
    """Tests for the main process_and_chunk_text function."""

    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_empty_and_whitespace_text(self, strategy):
        """Test that empty or whitespace-only text results in an empty list for all strategies."""
        options = TextProcessingOptions(chunking_strategy=strategy)
        assert process_and_chunk_text("", options) == []
        assert process_and_chunk_text("   ", options) == []
        assert process_and_chunk_text(" \n\n\t ", options) == []

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
        assert process_and_chunk_text(text, options) == expected

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
        assert process_and_chunk_text(text, options) == expected

    def test_balanced_strategy_properties(self):
        """Test the BALANCED strategy, focusing on properties rather than exact output."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=150,
            ideal_chunk_length=100,
            to_lowercase=True
        )
        # Use the long paragraph text which requires complex splitting
        chunks = process_and_chunk_text(LONG_PARAGRAPH_TEXT, options)

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
        assert process_and_chunk_text(PARAGRAPH_TEXT, options) == expected

    def test_paragraph_strategy_long_paragraph_fallback(self):
        """Test PARAGRAPH strategy's fallback to BALANCED for a long paragraph."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=120,
            ideal_chunk_length=100,
            to_lowercase=True
        )
        # This text has one paragraph that is much longer than max_chunk_length
        chunks = process_and_chunk_text(LONG_PARAGRAPH_TEXT, options)
        
        # It should have been split into multiple chunks by the balanced strategy
        assert len(chunks) > 1
        assert all(len(c) <= options.max_chunk_length for c in chunks)

    def test_option_to_lowercase(self):
        """Verify the to_lowercase option works as expected."""
        text = "This Is Mixed CASE."
        # Test when True
        options_true = TextProcessingOptions(to_lowercase=True)
        assert process_and_chunk_text(text, options_true) == ["this is mixed case."]
        
        # Test when False
        options_false = TextProcessingOptions(to_lowercase=False)
        assert process_and_chunk_text(text, options_false) == ["This Is Mixed CASE."]

    def test_option_remove_bracketed_text(self):
        """Verify the remove_bracketed_text option works as expected."""
        # Test when True
        options_true = TextProcessingOptions(remove_bracketed_text=True, to_lowercase=False)

        # Using the default PARAGRAPH strategy, this is returned as a single chunk.
        expected_true = ['This is useful text and . .']
        assert process_and_chunk_text(TEXT_WITH_BRACKETS, options_true) == expected_true

        # Test when False (this part of the original test was correct)
        options_false = TextProcessingOptions(remove_bracketed_text=False, to_lowercase=False)
        expected_false = [TEXT_WITH_BRACKETS]
        assert process_and_chunk_text(TEXT_WITH_BRACKETS, options_false) == expected_false

    def test_pre_splitting_of_very_long_sentences(self):
        """Ensure sentences longer than max_chunk_length are pre-split."""
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.GREEDY, # A strategy that uses _prepare_sentences
            max_chunk_length=100,
        )
        chunks = process_and_chunk_text(VERY_LONG_SENTENCE, options)

        assert len(chunks) > 1, "The single long sentence should have been split"
        for chunk in chunks:
            assert len(chunk) <= options.max_chunk_length

    def test_unknown_strategy_raises_error(self):
        """Test that an invalid strategy raises a ValueError."""
        # This bypasses Pydantic's enum validation for testing purposes
        options = TextProcessingOptions()
        options.chunking_strategy = "nonexistent_strategy"

        with pytest.raises(ValueError, match="Unknown chunking strategy: 'nonexistent_strategy'"):
            process_and_chunk_text("some text", options)

    @patch('tts_api.utils.text_processing.get_segmenter')
    def test_language_option_is_passed_to_segmenter(self, mock_get_segmenter):
        """Check if the text_language option is correctly passed to the segmenter factory."""
        # Mock the segmenter to control its output and avoid downloading models
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = ["Ein Satz"]
        mock_get_segmenter.return_value = mock_segmenter
        
        # Use a strategy that calls _prepare_sentences, like BALANCED
        options = TextProcessingOptions(
            text_language="de",
            chunking_strategy=TextChunkingStrategy.BALANCED
        )
        
        process_and_chunk_text("Ein Satz.", options)
        
        # The key assertion: was the factory called with the correct language?
        mock_get_segmenter.assert_called_once_with("de")


class TestProcessAndChunkTextEdgeCases:
    """Probes the boundaries, interactions, and corner cases of the text processing."""

    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_text_with_only_removable_content(self, strategy):
        """Test that text becoming empty after cleaning results in an empty list."""
        text = "[all content is inside brackets] (*and parentheses*)"
        options = TextProcessingOptions(
            chunking_strategy=strategy,
            remove_bracketed_text=True
        )
        assert process_and_chunk_text(text, options) == []

    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_unicode_and_emojis_are_handled(self, strategy):
        """Ensure non-ASCII characters don't break the logic and lengths are correct."""
        text = "你好, 世界. This is a very long test sentence to ensure a split happens 😊."
        options = TextProcessingOptions(
            chunking_strategy=strategy,
            max_chunk_length=55,
            to_lowercase=False
        )
        chunks = process_and_chunk_text(text, options)
        reconstructed_text = " ".join(chunks)
        assert "你好, 世界" in reconstructed_text
        assert "😊" in reconstructed_text
        assert all(len(c) <= 55 for c in chunks)

    def test_multiple_consecutive_newlines_in_paragraph_strategy(self):
        """PARAGRAPH strategy should handle multiple newlines as a single separator."""
        text = "Paragraph 1.\n\n\n\nParagraph 2."
        options = TextProcessingOptions(chunking_strategy=TextChunkingStrategy.PARAGRAPH)
        chunks = process_and_chunk_text(text, options)
        assert chunks == ["paragraph 1.", "paragraph 2."]

    @patch('tts_api.utils.text_processing._chunk_sentences_balanced')
    def test_paragraph_boundary_conditions_with_max_length(self, mock_balanced_chunker):
        """Test PARAGRAPH strategy's fallback logic right at the max_length boundary."""
        # Set a mock return value to prevent errors if it's called.
        mock_balanced_chunker.return_value = ["mocked", "split"]
        
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.PARAGRAPH,
            max_chunk_length=52,
            to_lowercase=False
        )
        
        # Case 1: Length is exactly max_length. Should NOT fall back.
        text_at_limit = "This paragraph has length of exactly fifty-two chars"
        assert len(text_at_limit) == 52
        chunks_at_limit = process_and_chunk_text(text_at_limit, options)
        assert chunks_at_limit == [text_at_limit]
        mock_balanced_chunker.assert_not_called() # Crucial check
        
        # Case 2: Length is one over max_length. SHOULD fall back.
        text_over_limit = "This paragraph has a length of over fifty-two characters."
        assert len(text_over_limit) > 52
        chunks_over_limit = process_and_chunk_text(text_over_limit, options)
        # We assert that the fallback was called and check its output.
        mock_balanced_chunker.assert_called_once()
        assert chunks_over_limit == ["mocked", "split"] # Check we got the mocked output

    def test_greedy_strategy_boundary_fill(self):
        """Test GREEDY strategy when a chunk exactly fills or just misses max_length."""
        s1 = "This sentence is a specific length for this boundary test." # 58 chars
        s2 = "Short one." # 10 chars
        text = f"{s1} {s2}" # Combined with space: 58 + 1 + 10 = 69 chars

        # Case 1: max_length allows the combination
        options_fit = TextProcessingOptions(chunking_strategy=TextChunkingStrategy.GREEDY, max_chunk_length=69)
        assert process_and_chunk_text(text, options_fit) == [f"{s1.lower()} {s2.lower()}"]

        # Case 2: max_length is one too short, forcing a split
        options_no_fit = TextProcessingOptions(chunking_strategy=TextChunkingStrategy.GREEDY, max_chunk_length=66)
        assert process_and_chunk_text(text, options_no_fit) == [s1.lower(), s2.lower()]

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
        chunks = process_and_chunk_text(text, options)
        assert chunks == ["Start. End."]

    def test_no_valid_split_points_in_long_sentence(self):
        """Test _force_split_long_chunk's hard cut when no delimiters are found."""
        long_word = "a" * 150
        options = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.GREEDY,
            max_chunk_length=100
        )
        chunks = process_and_chunk_text(long_word, options)
        assert chunks == ["a" * 100, "a" * 50]
        
    @pytest.mark.parametrize("strategy", list(TextChunkingStrategy))
    def test_chunks_are_never_empty_strings(self, strategy):
        """Ensure no strategy ever returns an empty string in the chunk list."""
        text = "Text...  More text.\n\n\nFinal text."
        # FIX: Changed max_chunk_length from 15 to a valid 50.
        options = TextProcessingOptions(chunking_strategy=strategy, max_chunk_length=50)
        
        chunks = process_and_chunk_text(text, options)
        
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
        # - Greedy would group S1+S2 (len 145) and leave S3 (len 10) as a tiny chunk.
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

        chunks = process_and_chunk_text(text, options)

        expected_chunks = [
            s1,
            f"{s2} {s3}"
        ]

        assert chunks == expected_chunks
        # Also check the greedy strategy to prove they behave differently
        options.chunking_strategy = TextChunkingStrategy.GREEDY
        greedy_chunks = process_and_chunk_text(text, options)
        assert greedy_chunks != expected_chunks
        assert greedy_chunks == [f"{s1} {s2}", s3]

    def test_avoids_greedy_choice_due_to_shortness_penalty2(self):
        """
        Tests the core value of the balanced strategy: balanced text chunks.
        """
        # Scenario:
        # - Greedy would group S1+S2 (len 145) and leave S3 (len 55) as another chunk.
        # - Balanced, with a moderate penalty, should prefer to split S1 (len 89)
        #   and group S2+S3 (len 110) to create two more evenly sized chunks.
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

        chunks = process_and_chunk_text(text, options)

        expected_chunks = [
            s1,
            f"{s2} {s3}"
        ]

        assert chunks == expected_chunks
        # Also check the greedy strategy to prove they behave differently
        options.chunking_strategy = TextChunkingStrategy.GREEDY
        greedy_chunks = process_and_chunk_text(text, options)
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
        chunks_a = process_and_chunk_text(text, options_a)

        # Scenario B: Larger ideal length should favor fewer, larger chunks.
        options_b = TextProcessingOptions(
            chunking_strategy=TextChunkingStrategy.BALANCED,
            max_chunk_length=200,
            ideal_chunk_length=150, # Large ideal length
            to_lowercase=False
        )
        chunks_b = process_and_chunk_text(text, options_b)

        # The primary assertion is that the choice of ideal_length had an effect.
        assert len(chunks_a) > len(chunks_b)
        assert len(chunks_a) == 4 # Should likely split into individual sentences
        assert len(chunks_b) == 2 # Should likely group sentences 1+2 and 3+4

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
            max_chunk_length=170, # Combined length with space is 182, so this is too small.
            ideal_chunk_length=200, # Ideal is high, encouraging large chunks.
            to_lowercase=False
        )

        chunks = process_and_chunk_text(text, options)

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
            to_lowercase=False
        )

        chunks = process_and_chunk_text(long_sentence_with_breaks, options)

        # It receives ['a'*70 + '.', 'b'*70 + '.', 'c'*70 + '.', 'd'*70 + '.']
        # It should group them in pairs.
        expected_chunks = [
            'a' * 70 + '. ' + 'b' * 70 + '.',
            'c' * 70 + '. ' + 'd' * 70 + '.',
        ]
        assert chunks == expected_chunks