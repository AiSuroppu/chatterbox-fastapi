# Pluggable TTS API

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular and highly configurable FastAPI wrapper for running open-source TTS engines. This project is designed to provide a production-ready, resilient API for a **single TTS engine at a time**, with a focus on quality, control, and robustness.

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

1.  Create a file named `request.json` with your desired parameters (see the full payload below).
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

### Full JSON Request Payload

This example shows all available parameters with detailed comments. You can selectively include only the ones you want to change from their defaults. For ground-truth definitions, see `src/tts_api/core/models.py`.

```jsonc
{
  // --- Core Generation Controls ---
  "text": "The text to be synthesized. This is the only required field.",
  "seed": 0, // Random seed for generation. Use 0 for a random seed, or a specific integer for reproducibility. Can be a list of seeds to use for each `best_of` candidate.
  "best_of": 1, // Generate this many candidates and return the best one. Increases quality at the cost of speed.
  "max_retries": 1, // For each chunk, retry generation this many times if it fails validation.

  // --- Engine-Specific Parameters ---
  // The name and content of this object changes depending on the engine.
  // This example is for 'chatterbox', a voice cloning engine.
  "chatterbox_params": {
    "exaggeration": 0.5,
    "temperature": 0.8,
    "cfg_weight": 0.5,
    "disable_watermark": true,
    "use_analyzer": false,
    "voice_cache_token": null // If you have a token, provide it here to bypass 'ref_audio' upload.
  },

  // --- Text Pre-Processing Controls ---
  "text_processing": {
    "text_language": "en", // Language for sentence tokenization and normalization.
    "to_lowercase": false,
    "remove_bracketed_text": false,
    "use_nemo_normalizer": false, // Use NeMo to expand numbers, dates, etc. into words.
    "apply_advanced_cleaning": false, // Heuristic cleaning (e.g., removes stutters).
    "chunking_strategy": "paragraph", // "paragraph", "balanced", "greedy", "simple".
    "shortness_penalty_factor": 2.0, // Penalizes short chunks in the 'balanced' strategy. Higher values avoid tiny chunks more aggressively.
    "ideal_chunk_length": 300, // Target character length for "balanced" strategy.
    "max_chunk_length": 500 // Hard character limit for any chunk.
  },

  // --- Audio Validation Controls ---
  // Configure rules to detect and discard low-quality audio. Set any value to `null` to disable that check.
  "validation": {
    "max_silence_dbfs": -60.0, // Fail if audio is quieter than this (dBFS).

    "max_contiguous_silence_s": 2.8, // Fail if there's a long silent gap in the middle.

    "min_voiced_duration_per_syllable": 0.150, // Global minimum duration per syllable to catch truncated audio.
    "max_voiced_duration_per_syllable": 0.350, // Maximum duration per syllable for *normal* words. Catches rambling.
    "min_syllables_for_duration_validation": 7, // Don't run any duration validation on very short text.
    "use_word_level_duration_analysis": true, // Enables a word-by-word analysis to handle mixed-complexity text. Highly recommended to leave this enabled.
    // The following parameters are only active when 'use_word_level_duration_analysis' is true.
    "min_word_len_for_low_complexity_analysis": 7, // A word must be this long to be considered for low-complexity analysis. The default is high to prevent misclassifying common English words.
    "low_complexity_log_variety_threshold": 1.6, // Words with a Log-Normalized Variety (LNV) below this are considered low-complexity. LNV = unique_chars / log(word_length).
    "low_complexity_max_duration_per_syllable": 0.4, // The relaxed maximum duration budget for words identified as low-complexity.
    "max_clipping_percentage": 0.1, // Fail if the audio has distortion.
    "min_spectral_centroid_std_dev": 1800.0, // Fail if the audio is monotonous (e.g., pure noise).
    "min_syllables_for_spectral_validation": 7 // Don't run spectral validation on very short text.
  },

  // --- Audio Post-Processing and Export Controls ---
  "post_processing": {
    // VAD-based Denoising and Trimming
    "denoise_with_vad": true,
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 100,
    "vad_min_speech_ms": 250,
    "vad_speech_pad_ms": 50, // Padding added to the start/end of detected speech.
    "vad_fade_ms": 10, // Duration of fade-in/out at speech boundaries for smooth transitions.
    "max_voiced_dynamic_range_db": 20.0, // Segments much quieter than the loudest will be filtered out.

    // Silence Insertion between Segments
    "trim_segment_silence": true, // Recommended to be true for custom silence.
    "inter_segment_silence_sentence_ms": 200, // Pause between sentences.
    "inter_segment_silence_paragraph_ms": 600, // Pause between paragraphs.
    "inter_segment_silence_fallback_ms": 150, // Pause between segments split by length, not semantics.
    
    // Final Silence Control
    "lead_in_silence_ms": null, // Desired silence at the start. `null` preserves original, `0` trims all.
    "lead_out_silence_ms": null, // Desired silence at the end. `null` preserves original, `0` trims all.

    // Loudness Normalization
    "normalize_audio": true,
    "normalize_method": "ebu", // "ebu" (for LUFS) or "peak".
    "normalize_level": -18.0, // Target LUFS for EBU, or dBFS for peak.

    // Export Format
    "export_format": "mp3" // "mp3", "wav", or "flac".
  }
}
```

## Configuration (`.env` file)

The `configure.py` script creates a `.env` file in the project root. You can edit this file to change server-wide settings.

-   `ENABLED_MODELS`: A list of engine names to load on startup. Should match the engine you configured.
    -   Example: `ENABLED_MODELS='["chatterbox"]'`
-   `DEVICE`: The compute device to use (`cuda` or `cpu`). This is set automatically by the configure script.
-   `CHATTERBOX_MAX_BATCH_SIZE`: Maximum text segments to process in a single batch. Increase on high-VRAM GPUs for better throughput.
-   `CHATTERBOX_COMPILE_MODE`: PyTorch 2.0+ model compilation mode (e.g., 'default', 'reduce-overhead'). Can significantly speed up inference after a one-time warm-up. Set to `""` to disable.
-   `CHATTERBOX_VOICE_CACHE_SIZE`: The maximum number of voice embeddings to keep in the server-side cache. Set to `0` to disable voice caching.
-   `CHATTERBOX_OFFLOAD_S3GEN`: (`True`/`False`) Offloads the main audio generation model (S3) to the CPU after use. Saves a small amount of VRAM at a negligible performance cost. Recommended for GPUs with less VRAM.
-   `CHATTERBOX_OFFLOAD_T3`: (`True`/`False`) Offloads the text-to-token model (T3) to the CPU after use.

## Extending the API (For Developers)

The project is designed to be truly pluggable. To add support for a new TTS engine (e.g., "my_engine"), follow these steps:

1.  **Add to `configure.py`**: Add the engine's name and Git repository to the `ENGINE_REGISTRY` in `configure.py` so the setup script can install it.
2.  **Create Engine Class**: In `src/tts_api/tts_engines/`, create a new file for your engine (e.g., `my_engine.py`). It must contain a class that inherits from `AbstractTTSEngine` and implements the `load_model`, `prepare_generation`, and `generate` methods. The `prepare_generation` method is where you will implement your engine's specific voice logic (e.g., handling `voice_id` or `ref_audio`).
3.  **Define Pydantic Models**: In `src/tts_api/core/models.py`, create two Pydantic models:
    *   A model for your engine's specific parameters (e.g., `MyEngineParams`).
    *   A final request model that inherits from `BaseTTSRequest` and includes your params model (e.g., `class MyEngineRequest(BaseTTSRequest): my_engine_params: MyEngineParams = Field(...)`).
4.  **Register the Engine in `main.py`**: This is the final step. Import your engine instance and request model, then add a new `EngineDefinition` to the central `ENGINE_REGISTRY` dictionary.

    ```python
    # In main.py
    ENGINE_REGISTRY = {
        "chatterbox": EngineDefinition(...),
        "my_engine": EngineDefinition(
            instance=my_engine_instance,
            request_model=MyEngineRequest,
            param_field_name="my_engine_params" # Must match the field name in MyEngineRequest
        )
    }
    ```

That's it! The API will automatically expose `/my_engine/generate` and `/engines/my_engine/config` endpoints without any further code changes.