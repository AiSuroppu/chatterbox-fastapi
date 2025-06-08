# TTS API

A modular FastAPI wrapper for running various open-source TTS engines. This project is designed to be configured to run a single TTS engine at a time.


## Supported Engines

The configuration script can set up the following engines:
-   `chatterbox`


## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AiSuroppu/chatterbox-fastapi
    cd chatterbox-fastapi
    ```

---

### Conda Workflow

2.  **Create and Activate a Conda Environment:**
    ```bash
    # Create an environment with a compatible Python version
    conda create -n chatterboxttsapi python=3.11.* -y

    # Activate it
    conda activate chatterboxttsapi
    ```

3.  **Run the Configure Script:**
    From within your activated environment, run `configure.py`. The script will auto-detect your CUDA version and install the correct GPU-enabled PyTorch build.

    ```bash
    # This will set up the 'chatterbox' engine
    python configure.py --engine chatterbox
    ```

    **Manual CUDA Version:** If auto-detection fails or you need a specific version (e.g., in a Docker container), use the `--cuda-version` flag.
    ```bash
    # Example for forcing CUDA 11.8
    python configure.py --engine chatterbox --cuda-version 11.8
    ```
    If no CUDA is found, it will install the CPU-only version of PyTorch and set the `DEVICE` accordingly in the `.env` file.

4.  **Run the Server:**
    Your environment is now ready. Start the FastAPI server:
    ```bash
    uvicorn main:app --host 127.0.0.1 --port 8000
    ```

---

### Alternative Workflow (with venv)

If you do not use Conda, the configure script will automatically create a local Python virtual environment in a `.venv/` directory.

2.  **Run the Configure Script:**
    ```bash
    python configure.py --engine chatterbox
    ```
    
    **Manual CUDA Version:** If auto-detection fails or you need a specific version (e.g., in a Docker container), use the `--cuda-version` flag.
    ```bash
    # Example for forcing CUDA 11.8
    python configure.py --engine chatterbox --cuda-version 11.8
    ```
    If no CUDA is found, it will install the CPU-only version of PyTorch and set the `DEVICE` accordingly in the `.env` file.

3.  **Activate the Environment and Run:**
    ```bash
    # On macOS/Linux
    source .venv/bin/activate

    # On Windows
    .\.venv\Scripts\activate
    ```
    Then, run the server:
    ```bash
    uvicorn main:app --host 127.0.0.1 --port 8000
    ```


## API Usage

The API accepts `multipart/form-data` requests. This means you must provide the JSON payload and the audio file as separate parts of the same request.

### Example Request with `curl`

This is the most common way to interact with the API from the command line.

1.  First, create a file named `request.json` with your desired parameters (see the full example below).
2.  Then, run the `curl` command:

```bash
curl -X POST "http://127.0.0.1:8000/chatterbox/generate" -F "req_json=<request.json" -F "ref_audio=@/path/to/your/reference_voice.wav" -o "output.mp3"
```
-   `-F "req_json=<request.json"`: Sends the content of `request.json` as the `req_json` form field.
-   `-F "ref_audio=@/path/to/your/reference_voice.wav"`: Uploads the specified audio file as the `ref_audio` form field.
-   `-o "output.mp3"`: Saves the streaming audio response to a file named `output.mp3`.

### Full JSON Request Payload (`request.json`)

This example shows all available parameters with their default values. You can create a file with this content to use with the `curl` command above. Check `src/tts_api/core/models.py` for details.

```jsonc
{
  "text": "This is a test of the text-to-speech generation. I can control many different parameters.",
  "seed": 0,
  "best_of": 1,
  "text_processing": {
    "to_lowercase": true,
    "remove_bracketed_text": false,
    "batching_strategy": "paragraph",
    "shortness_penalty_factor": 2.0,
    "ideal_chunk_length": 300,
    "max_chunk_length": 500
  },
  "post_processing": {
    "denoise_with_vad": true,
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 100,
    "vad_speech_pad_ms": 50,
    "vad_fade_ms": 10,
    "normalize_audio": true,
    "normalize_method": "ebu",
    "normalize_level": -18.0,
    "export_format": "mp3"
  },
  "chatterbox_params": {
    "exaggeration": 0.5,
    "temperature": 0.8,
    "cfg_weight": 0.5,
    "disable_watermark": true
  }
}
```

#### Parameter Groups Explained:

-   **`text`**: (string, required) The text to be synthesized.
-   **`seed`**: (integer) The random seed for generation. Use `0` for a different result each time.
-   **`best_of`**: (integer) Generate multiple speech outputs and automatically return the one with the highest speech-to-audio-duration ratio. Higher values take longer but can improve quality.
-   **`text_processing`**: Controls how the input text is cleaned and split into chunks before being sent to the TTS model.
-   **`post_processing`**: Controls audio effects applied *after* the raw audio is generated, such as silence removal and loudness normalization.
-   **`chatterbox_params`**: (object) Parameters that are passed directly to the Chatterbox TTS engine to control the generation process itself. This object will change if you are using a different engine.
