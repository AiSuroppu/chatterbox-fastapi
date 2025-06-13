import pytest
import torch
import io
import logging
from unittest.mock import patch, MagicMock, call, ANY
from collections import Counter

# Import the new Enum to use in tests
from tts_api.services.tts_service import generate_speech_from_request
from tts_api.core.models import ChatterboxTTSRequest
from tts_api.services.validation import ValidationResult
from tts_api.tts_engines.base import AbstractTTSEngine

# A dummy tensor to represent successful audio generation
SUCCESSFUL_WAVEFORM = torch.tensor([[1., 2., 3.]], dtype=torch.float32)

@pytest.fixture
def mock_engine():
    """Creates a mock TTS engine instance."""
    engine = MagicMock(spec=AbstractTTSEngine)
    engine.sample_rate = 24000
    # Default behavior: successfully generate one waveform for one text chunk
    engine.generate.return_value = [SUCCESSFUL_WAVEFORM]
    return engine

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for tts_service.py."""
    # Use a dictionary to hold all mocks for easy access in tests
    mocks = {
        'process_and_chunk_text': mocker.patch('tts_api.services.tts_service.process_and_chunk_text', return_value=['Hello world.']),
        'post_process_audio': mocker.patch('tts_api.services.tts_service.post_process_audio', return_value=io.BytesIO(b'final_audio')),
        'get_speech_ratio': mocker.patch('tts_api.services.tts_service.get_speech_ratio', return_value=0.9),
        'set_seed': mocker.patch('tts_api.services.tts_service.set_seed'),
        'torch_cat': mocker.patch('torch.cat', return_value=SUCCESSFUL_WAVEFORM),
        'tempfile': mocker.patch('tts_api.services.tts_service.tempfile'),
        'os': mocker.patch('tts_api.services.tts_service.os'),
        'random_randint': mocker.patch('random.randint'),
    }

    # Mock the validator to always pass by default
    mock_validator = MagicMock()
    mock_validator.is_valid.return_value = ValidationResult(is_ok=True)
    mocker.patch('tts_api.services.tts_service.ALL_VALIDATORS', [mock_validator])
    mocks['mock_validator'] = mock_validator

    # Mock tempfile to avoid actual file I/O
    mock_file_context = MagicMock()
    mock_file_context.name = "/tmp/fake_ref.wav"
    mock_file_context.__enter__.return_value = mock_file_context
    mocks['tempfile'].NamedTemporaryFile.return_value = mock_file_context
    mocks['os'].path.exists.return_value = True

    # Provide a long list of predictable "random" numbers
    mocks['random_randint'].side_effect = list(range(1000, 90000, 1000))

    return mocks


class TestGenerateSpeechFromRequest:
    """Test suite for the main TTS service logic."""

    def test_simple_generation_success(self, mock_dependencies, mock_engine):
        """
        Tests a basic, successful generation with one candidate and no retries.
        """
        req = ChatterboxTTSRequest(text="Hello world.", seed=42)

        result_buffer = generate_speech_from_request(req, mock_engine, req.chatterbox_params)

        mock_dependencies['process_and_chunk_text'].assert_called_once_with(text="Hello world.", options=req.text_processing)
        # Seed is set for the single candidate group
        mock_dependencies['set_seed'].assert_called_once_with(42)
        mock_engine.generate.assert_called_once_with(['Hello world.'], req.chatterbox_params, ref_audio_path=None)
        mock_dependencies['torch_cat'].assert_called_once_with([SUCCESSFUL_WAVEFORM], dim=1)
        mock_dependencies['post_process_audio'].assert_called_once_with(SUCCESSFUL_WAVEFORM, mock_engine.sample_rate, req.post_processing)
        assert result_buffer.read() == b'final_audio'

    def test_best_of_selects_highest_score(self, mock_dependencies, mock_engine):
        """
        Tests that `best_of` generates multiple candidates and selects the one with the best score.
        """
        req = ChatterboxTTSRequest(text="Find the best one.", best_of=3, seed=42)
        
        # Mock waveforms for each candidate
        wave_A = torch.tensor([[1.]])
        wave_B = torch.tensor([[2.]])
        wave_C = torch.tensor([[3.]])
        # Each generate call corresponds to one candidate with a unique seed
        mock_engine.generate.side_effect = [[wave_A], [wave_B], [wave_C]]
        
        # Mock scores to make the second candidate the best
        mock_dependencies['get_speech_ratio'].side_effect = [0.8, 0.95, 0.7]

        generate_speech_from_request(req, mock_engine, req.chatterbox_params)

        # Check that 3 candidates were generated
        assert mock_engine.generate.call_count == 3
        # The seeds should be deterministic and unique for each candidate
        mock_dependencies['set_seed'].assert_has_calls([call(42), call(1042), call(2042)], any_order=True)

        # Check that the waveform with the highest score was chosen
        mock_dependencies['torch_cat'].assert_called_once_with([wave_B], dim=1)

    def test_retries_on_engine_failure(self, mock_dependencies, mock_engine, caplog):
        """
        Tests that the service retries generation if the engine returns None.
        """
        req = ChatterboxTTSRequest(text="This will fail once.", max_retries=1, seed=42)
        
        # First call fails (returns None), second succeeds
        mock_engine.generate.side_effect = [[None], [SUCCESSFUL_WAVEFORM]]

        with caplog.at_level(logging.WARNING):
            generate_speech_from_request(req, mock_engine, req.chatterbox_params)
            assert "Validation failed for chunk" in caplog.text
            assert "Engine returned empty/None output" in caplog.text

        # Initial attempt + 1 retry
        assert mock_engine.generate.call_count == 2
        
        # First attempt uses the initial seed, the retry uses a new random seed for that candidate
        mock_dependencies['set_seed'].assert_has_calls([
            call(42),          # Initial attempt
            call(1000)         # Retry attempt (from mocked random.randint)
        ])
        mock_dependencies['torch_cat'].assert_called_once_with([SUCCESSFUL_WAVEFORM], dim=1)

    def test_retries_on_validation_failure(self, mock_dependencies, mock_engine, caplog):
        """
        Tests that the service retries if post-generation validation fails.
        """
        req = ChatterboxTTSRequest(text="This is invalid at first.", max_retries=1, seed=42)
        
        # Mock the validator to fail on the first check
        mock_validator = mock_dependencies['mock_validator']
        mock_validator.is_valid.side_effect = [
            ValidationResult(is_ok=False, reason="Audio is too short"),
            ValidationResult(is_ok=True)
        ]

        with caplog.at_level(logging.WARNING):
            generate_speech_from_request(req, mock_engine, req.chatterbox_params)
            assert "Validation failed for chunk" in caplog.text
            assert "Audio is too short" in caplog.text

        # Initial attempt + 1 retry
        assert mock_engine.generate.call_count == 2
        
        # First attempt uses the initial seed, the retry uses a new random seed for that candidate
        mock_dependencies['set_seed'].assert_has_calls([
            call(42),          # Initial attempt
            call(1000)         # Retry attempt (from mocked random.randint)
        ])
        mock_dependencies['torch_cat'].assert_called_once()


    def test_total_failure_raises_error(self, mock_dependencies, mock_engine, caplog):
        """
        Tests that a ValueError is raised if a chunk fails all its retry attempts.
        """
        req = ChatterboxTTSRequest(text="This will always fail.", max_retries=2, seed=42)
        
        # Engine always returns None, causing validation to fail
        mock_engine.generate.return_value = [None]

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="TTS failed to produce any valid candidates for segment 1"):
                generate_speech_from_request(req, mock_engine, req.chatterbox_params)
            assert "1 jobs failed all 3 attempts and will be excluded." in caplog.text

        # Initial attempt + 2 retries. Because it's one candidate, this is 3 calls.
        assert mock_engine.generate.call_count == 3
    
    def test_batching_by_candidate(self, mock_dependencies, mock_engine):
        """
        Tests that all segments for a given candidate are batched into a single engine call.
        """
        text_chunks = ['First part.', 'Second part.']
        mock_dependencies['process_and_chunk_text'].return_value = text_chunks
        
        req = ChatterboxTTSRequest(text="...", best_of=2, seed=[42, 99])

        wave_cand1 = [torch.tensor([[42.]]), torch.tensor([[42.]])]
        wave_cand2 = [torch.tensor([[99.]]), torch.tensor([[99.]])]
        mock_engine.generate.side_effect = [wave_cand1, wave_cand2]

        mock_dependencies['get_speech_ratio'].side_effect = [0.9, 0.9, 0.8, 0.8]

        generate_speech_from_request(req, mock_engine, req.chatterbox_params)

        # There are 2 candidates, so exactly 2 calls to generate, one for each seed
        assert mock_engine.generate.call_count == 2
        
        # Check that the seeds were set correctly for each batch
        mock_dependencies['set_seed'].assert_has_calls([call(42), call(99)], any_order=True)
        
        # Check that both calls were batched correctly with all text segments
        mock_engine.generate.assert_has_calls([
            call(text_chunks, req.chatterbox_params, ref_audio_path=None),
            call(text_chunks, req.chatterbox_params, ref_audio_path=None)
        ], any_order=True)

        # Verify final assembly used the best candidate's waveforms (cand1)
        mock_dependencies['torch_cat'].assert_called_once_with(wave_cand1, dim=1)

    def test_seed_handling_list_overflow_and_zero(self, mock_dependencies, mock_engine, caplog):
        """
        Tests seed handling with a list that is shorter than `best_of` and contains a zero.
        """
        req = ChatterboxTTSRequest(text=".", best_of=4, seed=[42, 0, 99])
        
        with caplog.at_level(logging.WARNING):
            generate_speech_from_request(req, mock_engine, req.chatterbox_params)
            assert "best_of (4+) is greater than the number of seeds provided (3)" in caplog.text
        
        # Check the sequence of seeds used for the 4 candidates
        mock_dependencies['set_seed'].assert_has_calls([
            call(42),      # From list[0]
            call(1000),    # From list[1] which is 0 (random)
            call(99),      # From list[2]
            call(2000)     # Overflow, uses random
        ], any_order=True)

    def test_ref_audio_file_is_created_and_cleaned_up(self, mock_dependencies, mock_engine):
        """
        Tests that a temporary file for reference audio is created and then deleted.
        """
        req = ChatterboxTTSRequest(text="Test with ref audio.")
        ref_audio_data = b'some_audio_data'
        
        mock_file_context = mock_dependencies['tempfile'].NamedTemporaryFile.return_value
        fake_path = mock_file_context.name

        generate_speech_from_request(req, mock_engine, req.chatterbox_params, ref_audio_data=ref_audio_data)

        # Check file creation and write
        mock_dependencies['tempfile'].NamedTemporaryFile.assert_called_once_with(suffix=".wav", delete=False)
        mock_file_context.write.assert_called_once_with(ref_audio_data)
        
        mock_engine.generate.assert_called_once_with(ANY, ANY, ref_audio_path=fake_path)

        mock_dependencies['os'].path.exists.assert_called_once_with(fake_path)
        mock_dependencies['os'].remove.assert_called_once_with(fake_path)

    def test_no_text_after_processing_raises_error(self, mock_dependencies, mock_engine):
        """
        Tests that a ValueError is raised if text processing results in an empty list.
        """
        mock_dependencies['process_and_chunk_text'].return_value = []
        req = ChatterboxTTSRequest(text="[...]") # Text that might be filtered out

        with pytest.raises(ValueError, match="No text to synthesize after processing."):
            generate_speech_from_request(req, mock_engine, req.chatterbox_params)

    def test_batching_of_initial_pass_and_retries(self, mock_dependencies, mock_engine, caplog):
        """
        Tests that failures across different candidates are retried with unique seeds,
        resulting in separate, per-candidate retry batches.
        """
        # 1. SETUP
        # Two text segments, one of which will fail validation initially.
        text_segments = ["This segment is fine.", "This segment is too short at first."]
        mock_dependencies['process_and_chunk_text'].return_value = text_segments

        # Request two candidates, with one retry allowed.
        req = ChatterboxTTSRequest(text="...", best_of=2, max_retries=1, seed=[42, 99])

        # Define good/bad waveforms to control validation.
        GOOD_WAVE = torch.tensor([[1., 2.]])
        BAD_WAVE = torch.tensor([[0.]])
        
        mock_validator = mock_dependencies['mock_validator']
        mock_validator.is_valid.side_effect = lambda w, *args: ValidationResult(is_ok=torch.equal(w, GOOD_WAVE), reason="Bad Wave")

        # Define engine outputs: initial passes fail on seg1, retry passes succeed.
        mock_engine.generate.side_effect = [
            # Initial Pass, Candidate 0 (seed 42): seg1 fails
            [GOOD_WAVE, BAD_WAVE],
            # Initial Pass, Candidate 1 (seed 99): seg1 fails
            [GOOD_WAVE, BAD_WAVE],
            # Retry Pass for Candidate 0's failed job (new seed 1000)
            [GOOD_WAVE],
            # Retry Pass for Candidate 1's failed job (new seed 2000)
            [GOOD_WAVE],
        ]

        # Define scores: candidate 1 will be slightly better.
        mock_dependencies['get_speech_ratio'].side_effect = [
            0.8, # Cand 0, seg0 (initial)
            0.9, # Cand 1, seg0 (initial)
            0.8, # Cand 0, seg1 (retry)
            0.9, # Cand 1, seg1 (retry)
        ]
        
        # 2. EXECUTION
        with caplog.at_level(logging.INFO):
            generate_speech_from_request(req, mock_engine, req.chatterbox_params)
            # 2 jobs fail validation (cand0/seg1 and cand1/seg1)
            assert caplog.text.count("Validation failed for chunk") == 2

        # 3. ASSERTIONS
        # We expect 4 calls: 2 for initial candidates, 2 for the separate retries.
        assert mock_engine.generate.call_count == 4
        
        # Verify seeds: 2 initial, 2 new seeds for retries.
        mock_dependencies['set_seed'].assert_has_calls(
            [call(42), call(99), call(1000), call(2000)], any_order=True
        )

        calls = mock_engine.generate.call_args_list
        failed_text = "This segment is too short at first."

        # Calls 0 & 1: Initial passes, full batch of segments
        assert calls[0].args[0] == text_segments
        assert calls[1].args[0] == text_segments
        
        # Calls 2 & 3: Retry passes, each with a single failed segment.
        assert calls[2].args[0] == [failed_text]
        assert calls[3].args[0] == [failed_text]

        # Final assembly should use the best complete candidate (cand 1, score 0.9)
        final_waveforms = mock_dependencies['torch_cat'].call_args[0][0]
        assert len(final_waveforms) == 2
        assert torch.equal(final_waveforms[0], GOOD_WAVE)
        assert torch.equal(final_waveforms[1], GOOD_WAVE)

    def test_complex_multi_pass_retry_and_batching(self, mock_dependencies, mock_engine, caplog):
            """
            Tests a complex multi-pass, multi-failure scenario to verify per-candidate
            seed and batching logic during retries.
            - 3 candidates, 5 segments, 3 retries.
            - Multiple, stateful failures across retries.
            - Verifies initial batching, per-candidate retry batching, and seed correctness.
            - Verifies final audio assembly by picking the best segment from all attempts.
            """
            # 1. SETUP: DEFINE THE SCENARIO
            req = ChatterboxTTSRequest(text="...", best_of=3, max_retries=3, seed=[42, 12])
            
            # Define the full, ordered sequence of seeds expected for all passes
            INITIAL_SEEDS = [42, 12, 1000] # 1000 is the first random number
            RETRY_1_SEEDS = [2000, 3000, 4000] # One new seed for each candidate's retries
            RETRY_2_SEEDS = [5000, 6000, 7000] # One new seed for each candidate's second retry
            FULL_SEED_SEQUENCE = INITIAL_SEEDS + RETRY_1_SEEDS + RETRY_2_SEEDS

            text_segments = [
                "s0_ok",
                "s1_fails_validation_once",
                "s2_fails_engine_once",
                "s3_fails_twice",
                "s4_ok",
            ]
            mock_dependencies['process_and_chunk_text'].return_value = text_segments

            # Create a unique waveform for every possible outcome. Each candidate's retry
            # will now produce a unique waveform.
            W = {f"c{c}_s{s}_p{p}": torch.tensor([[float(c*100 + s*10 + p)]]) 
                for c in range(3) for s in range(5) for p in range(3)} # Passes 0, 1, 2

            VALIDATION_FAIL_WAVEFORM = torch.tensor([[-1.]])
            ENGINE_FAIL_WAVEFORM = None

            # Create a stateful class to manage the mock's complex response logic based on call order.
            class MockTTSState:
                def __init__(self):
                    self.call_count = 0
                
                def __call__(self, texts, params, **kwargs):
                    self.call_count += 1
                    results = []

                    # Determine pass type and candidate index from the strict call order
                    if 1 <= self.call_count <= 3:   # Initial passes for candidates 0, 1, 2
                        pass_type = 0
                        cand_idx = self.call_count - 1
                    elif 4 <= self.call_count <= 6:  # Retry 1 passes for candidates 0, 1, 2
                        pass_type = 1
                        cand_idx = self.call_count - 4
                    else:  # Retry 2 passes for candidates 0, 1, 2
                        pass_type = 2
                        cand_idx = self.call_count - 7

                    for text in texts:
                        seg_idx = int(text[1]) # e.g., "s1_..." -> 1
                        if pass_type == 0: # Initial pass
                            if "s1_fails_validation_once" in text: results.append(VALIDATION_FAIL_WAVEFORM)
                            elif "s2_fails_engine_once" in text: results.append(ENGINE_FAIL_WAVEFORM)
                            elif "s3_fails_twice" in text: results.append(VALIDATION_FAIL_WAVEFORM)
                            else: results.append(W[f"c{cand_idx}_s{seg_idx}_p0"])
                        elif pass_type == 1: # First Retry
                            if "s3_fails_twice" in text: results.append(ENGINE_FAIL_WAVEFORM)
                            else: results.append(W[f"c{cand_idx}_s{seg_idx}_p1"])
                        elif pass_type == 2: # Second Retry
                            results.append(W[f"c{cand_idx}_s{seg_idx}_p2"])
                    return results

            mock_engine.generate.side_effect = MockTTSState()

            # Setup stateful mocks for validation and scoring
            mock_dependencies['mock_validator'].is_valid.side_effect = lambda w, *args: ValidationResult(
                is_ok=not torch.equal(w, VALIDATION_FAIL_WAVEFORM), reason="Intentional validation fail")
            
            # The score is the raw tensor value. This makes higher-numbered candidates/waves "better".
            mock_dependencies['get_speech_ratio'].side_effect = lambda w, *args: w.item()

            # 2. EXECUTION
            with caplog.at_level(logging.WARNING):
                generate_speech_from_request(req, mock_engine, req.chatterbox_params)

            # 3. ASSERTIONS
            
            # --- A. Verify Seed Handling (in strict order) ---
            mock_dependencies['set_seed'].assert_has_calls([call(s) for s in FULL_SEED_SEQUENCE], any_order=False)

            # --- B. Verify Batching and Call Count ---
            assert mock_engine.generate.call_count == 9 # 3 initial + 3 retry1 + 3 retry2
            calls = mock_engine.generate.call_args_list

            # Initial passes (calls 0-2): 3 calls, each with a full batch of 5 segments
            for i in range(3): assert calls[i].args[0] == text_segments
            
            # Retry 1 passes (calls 3-5): 3 calls, each batched with that candidate's 3 failed segments
            retry_1_texts = ["s1_fails_validation_once", "s2_fails_engine_once", "s3_fails_twice"]
            for i in range(3, 6): assert sorted(calls[i].args[0]) == sorted(retry_1_texts)

            # Retry 2 passes (calls 6-8): 3 calls, each batched with the 1 remaining failure
            retry_2_texts = ["s3_fails_twice"]
            for i in range(6, 9): assert calls[i].args[0] == retry_2_texts

            # --- C. Verify Logging ---
            assert caplog.text.count("Intentional validation fail") == 6 # s1 and s3 for 3 cands
            assert caplog.text.count("Engine returned empty/None output") == 6 # s2 for 3 cands, s3 for 3 cands on retry
            
            # --- D. Verify Final Audio Assembly ---
            # The service picks the best generation PER-SEGMENT. Since cand 2's scores
            # are always highest, its successful waveforms will be chosen.
            expected_final_waveforms = [
                W["c2_s0_p0"],      # s0: Cand 2 wins on initial pass
                W["c2_s1_p1"],      # s1: Cand 2 wins on first retry
                W["c2_s2_p1"],      # s2: Cand 2 wins on first retry
                W["c2_s3_p2"],      # s3: Cand 2 wins on second retry
                W["c2_s4_p0"],      # s4: Cand 2 wins on initial pass
            ]

            final_waveforms = mock_dependencies['torch_cat'].call_args.args[0]
            assert len(final_waveforms) == 5
            for i, (actual, expected) in enumerate(zip(final_waveforms, expected_final_waveforms)):
                assert torch.equal(actual, expected), f"Waveform for segment {i} is incorrect"