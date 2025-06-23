# Pluggable TTS API

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular and highly configurable FastAPI wrapper for running open-source TTS engines. This project is designed to provide a production-ready, resilient API for a **single TTS engine at a time**, with a focus on quality, control, and robustness.

## Full JSON Request Payload

This example shows all available parameters with detailed comments. You can selectively include only the ones you want to change from their defaults. For ground-truth definitions, see `src/tts_api/core/models.py`.

```jsonc
{
  // This document outlines the complete JSON request payload for the
  // Pluggable TTS API.
  // This JSON object should be sent as a string within a
  // 'multipart/form-data' field named 'req_json'.
  //
  // For voice cloning, a reference audio file should be sent in a separate
  // form field named 'ref_audio'.
  //
  // The structure is divided into logical groups:
  // 1. Core Generation Settings: Top-level controls for the generation
  //    process.
  // 2. Text Processing: How the input text is cleaned, normalized, and
  //    split into chunks.
  // 3. Validation: Rules to ensure the quality of each generated audio
  //    chunk before it's accepted.
  // 4. Post-Processing: How accepted audio chunks are assembled, enhanced,
  //    and encoded into the final file.
  // 5. Engine-Specific Parameters: Settings unique to the selected TTS
  //    engine (e.g., 'chatterbox_params').
  //
  // This example uses the 'chatterbox' engine. To use another engine like
  // 'fish_speech', replace the 'chatterbox_params' object with
  // 'fish_speech_params' and its corresponding settings.

  // -------------------------------------------
  // --- 1. Core Generation Settings         ---
  // -------------------------------------------
  // Top-level controls governing the overall synthesis process, including
  // text input, randomness, retries, and quality-improvement strategies.

  "text": "Your text to be converted to speech goes here.",
  // The text to be converted to speech. This is the only required field.

  "seed": 0,
  // Random seed for reproducibility. If a single integer is provided, it's
  // used as the base for all generations. A list of integers can be
  // provided (e.g., [123, 456]) to set a specific seed for each 'best_of'
  // candidate. A seed of 0 means a random seed will be used for each
  // generation, making results non-deterministic.

  "best_of": 1,
  // Generates this many audio candidates for each text chunk and
  // automatically returns the one with the highest speech-to-audio-duration
  // ratio (i.e., the least silence/noise). A value > 1 can significantly
  // improve quality and robustness at the cost of longer processing time.
  // (Range: 1-10)

  "max_retries": 1,
  // For each text chunk, if a generation attempt fails validation (e.g.,
  // produces silence, distorted audio, or is truncated), the API will
  // retry the generation this many times with a new random seed.
  // (Range: 0-10)

  // -------------------------------------------
  // --- 2. Text Processing                  ---
  // -------------------------------------------
  // Controls for how the raw input text is pre-processed before being sent
  // to the TTS model. This includes cleaning, normalization, and splitting
  // the text into manageable chunks for generation.
  "text_processing": {
    "text_language": "en",
    // Language code (e.g., 'en', 'de', 'es') used for language-specific
    // text processing like sentence segmentation and number normalization.

    "to_lowercase": false,
    // If true, converts all input text to lowercase before processing.

    "remove_bracketed_text": false,
    // If true, removes text inside brackets [] and parentheses ().
    // Useful for stripping annotations.

    "use_nemo_normalizer": false,
    // If true, uses the NeMo text normalizer to convert numbers, dates,
    // currencies, etc., into their full word forms (e.g., '$10.50' becomes
    // 'ten dollars and fifty cents'). Requires the 'en' language.

    "apply_advanced_cleaning": false,
    // If true, applies advanced, heuristic cleaning rules. This includes
    // removing stutters (e.g., 'b-but'), normalizing emphasis markers
    // (e.g., '*word*'), and other common text artifacts.

    "chunking_strategy": "paragraph",
    // Method to split input text into smaller chunks for the TTS engine.
    // 'paragraph': Keeps paragraphs whole. Long paragraphs are split using
    //              the 'balanced' strategy. (Recommended)
    // 'balanced':  Optimally splits sentences to create chunks of similar,
    //              ideal length.
    // 'greedy':    Quickly groups sentences into chunks without balancing.
    // 'simple':    Force-splits text by character length, ignoring sentence
    //              boundaries.

    "shortness_penalty_factor": 2.0,
    // Used by 'balanced' chunking. A higher value more aggressively
    // penalizes very short chunks, preferring more uneven but longer chunks.
    // (Min: 1.0)

    "ideal_chunk_length": 300,
    // The target character length for chunks when using the 'balanced'
    // strategy. (Range: 50-1000)

    "max_chunk_length": 500
    // The absolute maximum character length for any single text chunk. Text
    // will be forcibly split if it exceeds this. (Range: 50-1000)
  },

  // -------------------------------------------
  // --- 3. Validation                       ---
  // -------------------------------------------
  // A pipeline of checks performed on each generated audio chunk. If a chunk
  // fails validation, it is discarded and a retry is attempted (up to
  // 'max_retries'). This is critical for catching model failure modes.
  "validation": {
    // --- Silence & Truncation Validation ---
    "max_silence_dbfs": -60.0,
    // Fails generation if the overall energy (RMS) of the raw audio is
    // below this level in dBFS. Catches silent outputs. Null to disable.

    "max_contiguous_silence_s": 2.8,
    // Fails if any silent gap *within* a chunk is longer than this.
    // Catches outputs where the model stops speaking midway. Null to disable.

    // --- Speech Duration Validation (Syllable-based) ---
    // Ensures generated speech has a reasonable length relative to the
    // input text's syllable count. Effective at catching truncated
    // (cut-off) or rambling (hallucinated) audio.
    "min_voiced_duration_per_syllable": 0.150,
    // Fails if speech duration is less than (this value * syllable_count).
    // Catches truncated audio. Set to null to disable. (Range: 0.0-2.0)

    "max_voiced_duration_per_syllable": 0.350,
    // Fails if speech duration is more than (this value * syllable_count).
    // Catches hallucinated audio. Set to null to disable. (Range: 0.0-5.0)

    "min_syllables_for_duration_validation": 7,
    // Disables duration-per-syllable checks if a chunk has fewer syllables
    // than this. Prevents failures on short phrases like 'Okay.' or 'Yes.'.
    // Set to null to always run.

    // --- Advanced Duration Analysis for Complex Words ---
    // Allows for long, drawn-out words (e.g., 'Ahhhhhh') without failing
    // validation.
    "use_word_level_duration_analysis": true,
    // If true, analyzes words to identify 'low-complexity' words (like
    // onomatopoeia), which get a more relaxed max duration budget.

    "min_word_len_for_low_complexity_analysis": 7,
    // A word must have at least this many characters to be considered for
    // low-complexity analysis.

    "low_complexity_log_variety_threshold": 1.6,
    // Threshold for identifying a low-complexity word. A lower value means
    // a word needs to be more repetitive to be classified as low-complexity.

    "low_complexity_max_duration_per_syllable": 0.600,
    // Relaxed maximum duration per syllable for low-complexity words.

    // --- Audio Quality Validation ---
    "max_clipping_percentage": 0.1,
    // Fails if more than this percentage of audio samples are clipped.
    // Catches distorted outputs. Set to null to disable.

    "min_spectral_centroid_std_dev": 1800.0,
    // Fails if the standard deviation of the spectral centroid is too low.
    // Catches monotonous, robotic, or noisy outputs. Null to disable.

    "min_syllables_for_spectral_validation": 7
    // Disables the spectral centroid check for chunks with few syllables.
    // Prevents failures on short, naturally monotonous phrases.
  },

  // -------------------------------------------
  // --- 4. Post-Processing                  ---
  // -------------------------------------------
  // Controls for how validated audio chunks are stitched together, cleaned,
  // normalized, and exported into the final audio file.
  "post_processing": {
    // --- VAD-based Denoising & Filtering ---
    "denoise_with_vad": true,
    // Master switch for Voice Activity Detection (VAD) to clean audio. VAD
    // identifies and keeps only the speech portions of the audio.

    "vad_threshold": 0.5,
    // Confidence threshold (0.0 to 1.0) for VAD to classify a segment as
    // speech.

    "vad_min_silence_ms": 100,
    // Min duration of silence (ms) between speech segments.

    "vad_min_speech_ms": 250,
    // Speech segments shorter than this duration (ms) are discarded.

    "vad_speech_pad_ms": 50,
    // Padding (ms) added to the start/end of each speech segment.

    "vad_fade_ms": 10,
    // Duration (ms) of fade-in/out applied to speech segments.

    "max_voiced_dynamic_range_db": 20.0,
    // After VAD, if a voiced segment's RMS is this many dB quieter than
    // the loudest segment, it is filtered out. Set to null to disable.

    // --- Silence & Pacing Control ---
    "trim_segment_silence": true,
    // If true, trims leading/trailing silence from each audio chunk before
    // they are stitched together.

    "inter_segment_silence_sentence_ms": 200,
    // Silence (ms) between chunks representing sentences in one paragraph.

    "inter_segment_silence_paragraph_ms": 600,
    // Silence (ms) between chunks representing paragraph breaks.

    "inter_segment_silence_fallback_ms": 150,
    // Fallback silence (ms) for chunks split by length, not by sentence.

    // --- Normalization & Final Touches ---
    "normalize_audio": true,
    // If true, normalizes the final audio loudness using ffmpeg.

    "normalize_method": "ebu",
    // Normalization method. 'ebu' (EBU R128) for consistent loudness or
    // 'peak' for a specific peak level.

    "normalize_level": -18.0,
    // Target loudness level. In LUFS for 'ebu', in dBFS for 'peak'.

    "lead_in_silence_ms": null,
    // Desired duration of silence (ms) at the start of the final file.
    // Null preserves natural silence; 0 trims it.

    "lead_out_silence_ms": null,
    // Desired duration of silence (ms) at the end of the final file.
    // Null preserves natural silence; 0 trims it.

    "export_format": "mp3"
    // Final audio format. Options: "mp3", "wav", "flac".
  },


  // -------------------------------------------
  // --- 5. Engine-Specific Parameters       ---
  // -------------------------------------------
  // Parameters unique to the 'chatterbox' TTS engine. The key of this
  // object MUST match the engine name in the URL (e.g., /chatterbox/generate).
  "chatterbox_params": {
    "exaggeration": 0.5,
    // Emotion exaggeration factor for the cloned voice. (Range: 0.0-2.0)

    "temperature": 0.8,
    // Generation temperature. Higher values -> more random speech.
    // (Range: 0.01-5.0)

    "cfg_weight": 0.5,
    // Classifier-Free Guidance weight. Influences how closely the
    // generation adheres to the prompt. (Range: 0.0-1.0)

    "disable_watermark": true,
    // If true, disables Chatterbox's built-in audio watermark.

    "use_analyzer": false,
    // If true, enables an experimental analyzer which can improve quality
    // but may fail on some inputs.

    "voice_cache_token": null
    // A token for a server-side cached voice. If provided, 'ref_audio' is
    // ignored. The server returns a 'voice_cache_token' in the response
    // header after a successful generation with new reference audio.
    // Re-using this token is much faster than re-uploading the same file.
  }

  // --- Example for 'fish_speech' engine ---
  // To use the 'fish_speech' engine, you would replace the
  // 'chatterbox_params' object above with this one:
  /*
  "fish_speech_params": {
    "temperature": 0.8,         // Generation temperature. (Range: 0.01-2.0)
    "top_p": 0.8,               // Top-P (nucleus) sampling. (Range: 0.0-1.0)
    "repetition_penalty": 1.1,  // Penalty for repeating. (Range: 1.0-2.0)
    "max_new_tokens": 1024,     // Max semantic tokens per chunk. (1-4096)
    "ref_text": null,           // The transcription of the reference audio.
    "voice_cache_token": null   // Token for a server-side cached voice.
  }
  */
}
```

## Prerequisites

-   Python 3.8+
-   Git
-   [NVIDIA Driver](https://www.nvidia.com/download/index.aspx) (for GPU support)
-   (Optional) [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management.

## Setup

The setup process is streamlined by the `configure.py` script, which prepares the environment, installs dependencies, and creates a default `.env` configuration file.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AiSuroppu/chatterbox-fastapi
    cd chatterbox-fastapi
    ```

2.  **Run the Configure Script:**
    Execute the script, specifying the TTS engine you want to set up.

    ```bash
    # This will set up the 'chatterbox' engine
    python configure.py --engine chatterbox
    ```

    The script will:
    -   Detect if you're in a Conda environment. If not, it will create a local Python virtual environment in the current directory `./.venv/`.
    -   Auto-detect your CUDA version and install the correct GPU-enabled PyTorch build.
    -   Install all required Python packages for the API and the chosen engine.
    -   Create a `.env` file with default settings.

    <details>
    <summary><strong>Manual CUDA Version / CPU-Only Setup</strong></summary>

    -   If auto-detection fails or you need a specific version (e.g., in a Docker container), use the `--cuda-version` flag:
        ```bash
        # Example for forcing CUDA 11.8
        python configure.py --engine chatterbox --cuda-version 11.8
        ```
    -   If no CUDA is detected or specified, it will install the CPU-only version of PyTorch and set the `DEVICE` accordingly in the `.env` file.
    </p>
    </details>

3.  **Activate the Environment (if not using Conda):**
    If you are not using an active Conda environment, the script will create a venv. You must activate it:
    ```bash
    # On macOS/Linux
    source .venv/bin/activate

    # On Windows
    .\.venv\Scripts\activate
    ```

## Running the Server

Once the setup is complete and your environment is active, start the FastAPI server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

You can now access the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Configuration (`.env` file)

The `configure.py` script creates a `.env` file in the project root. You can edit this file to change server-wide settings.

## How It Works: The Generation Pipeline

Understanding the request lifecycle helps in using the API parameters effectively:

1.  **Text Processing**: The input text is cleaned and normalized based on `text_processing` settings.
2.  **Chunking**: The cleaned text is split into smaller, manageable chunks suitable for the TTS model based on `text_processing` settings.
3.  **Generation Loop**: For each chunk:
    -   The API generates `best_of` audio candidates.
    -   Each candidate is passed through the validation pipeline according to `validation` settings.
    -   If a candidate fails validation, it is retried up to `max_retries` times with a new random seed.
    -   The best valid candidate (highest quality score) is selected.
4.  **Assembly**: The best audio chunks are assembled, with silence inserted between them according to `post_processing` rules.
5.  **Final Post-Processing**: The complete audio is processed (VAD denoise, loudness normalization, etc.) and encoded into the desired `export_format` according to `post_processing` settings.

## API Usage

The API accepts `multipart/form-data` requests. This requires sending the JSON payload and any required files (like a reference voice) as separate parts of the same request.

### Discovering Engine Parameters

The request payload can be complex. To make it easy to see all available options and their default values for a specific engine, you can use the `/config` endpoint:

```bash
curl http://127.0.0.1:8000/engines/chatterbox/config | jq
```

This will return a complete JSON object that you can use as a template for your `/{engine_name}/generate` requests.

### Voice Caching and Management

Many TTS engines require an expensive "voice embedding" calculation. This embedding, which defines the voice's characteristics, can be created in several ways that benefit from caching:

*   **Voice Cloning:** Analyzing a user-provided `ref_audio` file to replicate the speaker's voice.
*   **Voice Synthesis:** Using a model to generate a unique voice based on a descriptive text prompt (e.g., "a clear, friendly female voice with a neutral American accent") or a list of predefined voice parameters.

In both cases, the initial computation is slow, but the resulting embedding can be cached by the server. The `voice_cache_token` is your key to using this cache. On any successful request that generates or uses a specific voice, the server will return the corresponding token in the `X-Generation-Metadata` response header.

**Workflow:**

1.  Make an initial generation request with a full **voice specification** (e.g., send a `ref_audio` file for cloning, or a `voice_prompt` for synthesis).
2.  If successful, the response will contain an `X-Generation-Metadata` header with a JSON payload like: `{"voice_cache_token": "some-unique-token"}`. **Store this token.**
3.  For all subsequent requests with the *same voice*, you can omit the original voice specification and instead include the token in your JSON payload:
    ```json
    {
      "chatterbox_params": {
        "voice_cache_token": "some-unique-token"
      }
    }
    ```
This speeds up requests and reduces bandwidth. The server cache size is configurable in the `.env` file.

#### Handling Expired Voice Tokens

If you provide a `voice_cache_token` that the server no longer has (because its cache expired or was cleared), the API will respond with an **`HTTP 409 Conflict`** error.

**To resolve this**, simply re-send your request with the original voice specification (e.g., the `ref_audio` file). A successful response will provide a new, valid `voice_cache_token` for you to use.

### Example with `curl`

1.  Create a file named `request.json` with your desired parameters (see the full payload above).
2.  Run the `curl` command, replacing the path to your reference audio. Use `-D -` to view the response headers and retrieve the token.

    ```bash
    curl -X POST "http://127.0.0.1:8000/chatterbox/generate" \
      -F "req_json=<request.json" \
      -F "ref_audio=@/path/to/your/reference_voice.wav" \
      -o "output.mp3" -D -
    ```

### Example with Python `requests`

```python
import requests
import json

# 1. Define the full request payload
payload = {
  "text": "This is a test of the text-to-speech generation. This API has many powerful features for robust audio creation.",
  "seed": 1234,
  "chatterbox_params": {
    "temperature": 0.75
  }
  # All other parameters will use their default values
}

# 2. Prepare the multipart/form-data
files = {
    'req_json': (None, json.dumps(payload), 'application/json'),
    'ref_audio': ('reference.wav', open('/path/to/your/reference_voice.wav', 'rb'), 'audio/wav')
}

# 3. Send the request
url = "http://127.0.0.1:8000/chatterbox/generate"
response = requests.post(url, files=files, stream=True)

# 4. Save the streaming audio response and check for metadata
if response.status_code == 200:
    # Check for the metadata header and save the token
    if "X-Generation-Metadata" in response.headers:
        metadata = json.loads(response.headers["X-Generation-Metadata"])
        voice_token = metadata.get("voice_cache_token")
        print(f"Received voice cache token: {voice_token}")
        # You should now save this token and use it for future requests

    with open("output.mp3", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Audio saved to output.mp3")
elif response.status_code == 409:
    print("Error: The voice token is invalid or expired.")
    print("Please re-run the request with the 'ref_audio' file to get a new token.")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## Extending the API (For Developers)

The project is designed to be truly pluggable. To add support for a new TTS engine (e.g., "my_engine"), follow these steps:

1.  **Add to `configure.py`**: Add the engine's name and Git repository to the `ENGINE_REGISTRY` in `configure.py` so the setup script can install it.
2.  **Create Engine Class**: In `src/tts_api/tts_engines/`, create a new file for your engine (e.g., `my_engine.py`). It must contain a class that inherits from `AbstractTTSEngine` and implements the `load_model`, `prepare_generation`, and `generate` methods. The `prepare_generation` method is where you will implement your engine's specific voice logic (e.g., handling `voice_id` or `ref_audio`).
3.  **Define Pydantic Models**: In `src/tts_api/core/models.py`, create two Pydantic models:
    *   A model for your engine's specific parameters (e.g., `MyEngineParams`).
    *   A final request model that inherits from `BaseTTSRequest` and includes your params model (e.g., `class MyEngineRequest(BaseTTSRequest): my_engine_params: MyEngineParams = Field(...)`).
4.  **Register the Engine in `main.py`**: This is the final step. Import your engine instance and request model, then add a new `EngineDefinition` to the central `SUPPORTED_ENGINES` dictionary.

    ```python
    # In main.py
    SUPPORTED_ENGINES = {
      "chatterbox": {
          "engine_path": "tts_api.tts_engines.chatterbox_engine.chatterbox_engine",
          "model_path": "tts_api.core.models.ChatterboxTTSRequest",
          "param_field_name": "chatterbox_params"
      },
      "fish_speech": {
          "engine_path": "tts_api.tts_engines.fish_speech_engine.fish_speech_engine",
          "model_path": "tts_api.core.models.FishSpeechTTSRequest",
          "param_field_name": "fish_speech_params"
      },
      # Add your engine here
    }
    ```

That's it! The API will automatically expose `/my_engine/generate` and `/engines/my_engine/config` endpoints without any further code changes.