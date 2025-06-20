import os
import subprocess
import sys
import argparse
import shutil
import re
from pathlib import Path

# A registry of known TTS engines and their git repositories
ENGINE_REGISTRY = {
    "chatterbox": {
        "repo": "https://github.com/AiSuroppu/chatterbox.git",
        "torch_version": "2.6.0",
        "torchaudio_version": "2.6.0",
        "setup_deps": [],
        "downloads": [],
    },
    "fish_speech": {
        "repo": "https://github.com/AiSuroppu/fish-speech.git",
        "torch_version": "2.6.0",
        "torchaudio_version": "2.6.0",
        "setup_deps": ["huggingface-hub"],
        "downloads": [
            {
                "type": "huggingface",
                "repo_id": "fishaudio/openaudio-s1-mini",
                "local_dir": "vendor/fish_speech/checkpoints/openaudio-s1-mini",
                "description": "FishSpeech main model checkpoints",
                "requires_login": True # Flag that this model is gated
            }
        ]
    }
}


def run_command(command, error_message):
    """Runs a command and exits on failure."""
    try:
        subprocess.run(command, check=True, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        print(f"Error: {error_message}")
        print(f"Command failed: {e.cmd}")
        sys.exit(1)

def detect_cuda_version():
    """Tries to detect the CUDA version using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.CalledProcessError, AttributeError):
        return None
    return None

def main(engine_name, cuda_version_override):
    if engine_name not in ENGINE_REGISTRY:
        print(f"Error: Unknown engine '{engine_name}'. Please choose from: {list(ENGINE_REGISTRY.keys())}")
        sys.exit(1)

    print(f"--- Configuring project for TTS engine: {engine_name} ---")
    engine_info = ENGINE_REGISTRY[engine_name]

    # --- Step 1: Prepare Python environment ---
    print("1. Preparing Python environment...")
    is_conda = 'CONDA_PREFIX' in os.environ
    venv_dir = ".venv"
    if is_conda:
        print(f"   - Detected active Conda environment: {os.environ.get('CONDA_DEFAULT_ENV')}")
        python_executable_path = sys.executable
    else:
        print("   - Creating local virtual environment in './.venv'.")
        if not os.path.exists(venv_dir):
            run_command(f'"{sys.executable}" -m venv {venv_dir}', "Failed to create venv.")
        python_executable_path = os.path.join(venv_dir, 'Scripts' if os.name == 'nt' else 'bin', 'python')

    pip_executable = f'"{python_executable_path}" -m pip'

    # --- Step 2: Clean up old vendor directory ---
    print("2. Cleaning up old vendor directory (if any)...")
    if os.path.isdir("vendor"):
        shutil.rmtree("vendor")
    
    # --- Step 3: Install PyTorch with correct CUDA support ---
    print("3. Installing PyTorch...")
    torch_version = engine_info["torch_version"]
    torchaudio_version = engine_info["torchaudio_version"]
    cuda_version = cuda_version_override or detect_cuda_version()
    torch_cmd = f"{pip_executable} install torch=={torch_version} torchaudio=={torchaudio_version}"
    if cuda_version:
        print(f"   - Found CUDA {cuda_version}. Installing PyTorch with CUDA support.")
        formatted_cuda = "cu" + cuda_version.replace('.', '')
        index_url = f"https://download.pytorch.org/whl/{formatted_cuda}"
        torch_cmd += f" --index-url {index_url}"
    else:
        print("   - No CUDA detected or specified. Installing CPU-only PyTorch.")
    run_command(torch_cmd, "Failed to install PyTorch.")

    # --- Step 4: Install the main TTS API wrapper project ---
    print("4. Installing the TTS API wrapper project in editable mode...")
    run_command(f"{pip_executable} install -e .", "Failed to install tts-api-wrapper project.")

    # --- Step 5: Clone the engine repository ---
    print(f"5. Cloning '{engine_name}' repository...")
    os.makedirs("vendor", exist_ok=True)
    submodule_path = os.path.join("vendor", engine_name).replace("\\", "/")
    run_command(f"git clone --depth 1 {engine_info['repo']} {submodule_path}", "Failed to clone engine repository.")
    
    # --- Step 6: Install the engine-specific dependencies ---
    print(f"6. Installing '{engine_name}' and its specific dependencies...")
    run_command(f"{pip_executable} install -e ./{submodule_path}", f"Failed to install {engine_name}.")
    
    # --- Step 7: Install setup-time dependencies for the engine ---
    snapshot_download = None
    if engine_info["setup_deps"]:
        print(f"7. Installing setup-time dependencies for '{engine_name}'...")
        deps_str = " ".join(engine_info["setup_deps"])
        run_command(f"{pip_executable} install {deps_str}", f"Failed to install setup dependencies for {engine_name}.")
        from huggingface_hub import snapshot_download
    else:
        print("7. No setup-time dependencies to install. Skipping.")
        
    # --- Step 8: Create .env file ---
    print("8. Creating .env file...")
    with open(".env", "w") as f:
        f.write(f'ENABLED_MODELS=\'["{engine_name}"]\'\n')
        f.write(f'DEVICE="{"cuda" if cuda_version else "cpu"}"\n\n')

        f.write("# NeMo text normalizer will cache compiled grammars here.\n")
        nemo_cache_path = os.path.join(os.getcwd(), ".cache", "nemo").replace("\\", "/")
        f.write(f'#NEMO_CACHE_DIR="{nemo_cache_path}"\n\n')

        f.write("# LRU cache sizes for language-specific normalizers and segmenters.\n")
        f.write("# Increase if you support many languages concurrently.\n")
        f.write("#NEMO_NORMALIZER_CACHE_SIZE=4\n")
        f.write("#PYSBD_CACHE_SIZE=4\n")
        f.write("# LRU cache size for the pyphen.Pyphen dictionary instances.\n")
        f.write("#PYPHEN_CACHE_SIZE=4\n\n")

        f.write("# --- Chatterbox Specific Settings ---\n")
        f.write("# The maximum number of text segments to process in a single batch for Chatterbox.\n")
        f.write("# Increase on high-VRAM GPUs for better throughput.\n")
        f.write("#CHATTERBOX_MAX_BATCH_SIZE=2\n")
        f.write("# PyTorch 2.0+ model compilation mode for Chatterbox (e.g., 'default', 'reduce-overhead', 'max-autotune').\n")
        f.write("# Can significantly speed up inference after a one-time warm-up cost.\n")
        f.write("# Set to an empty string to disable compilation.\n")
        f.write('CHATTERBOX_COMPILE_MODE=""\n\n')
        f.write("# The maximum number of voice embeddings to keep in the server-side cache.\n")
        f.write('#CHATTERBOX_VOICE_CACHE_SIZE=10\n')
        f.write("# Offload the S3 generation model to CPU after use to save VRAM. Set to False if you have abundant VRAM.\n")
        f.write("#CHATTERBOX_OFFLOAD_S3GEN=False\n")
        f.write("# Offload the T3 (text-to-token) model to CPU after use to save VRAM. Set to False if you have abundant VRAM.\n")
        f.write("#CHATTERBOX_OFFLOAD_T3=False\n\n")

        f.write("# --- FishSpeech Specific Settings ---\n")
        f.write("# NOTE: The model path is relative to the project root and is set by the download script.\n")
        f.write('#FISHSPEECH_T2S_CHECKPOINT_PATH="vendor/fish_speech/checkpoints/openaudio-s1-mini"\n')
        f.write('#FISHSPEECH_DECODER_CHECKPOINT_PATH="vendor/fish_speech/checkpoints/openaudio-s1-mini/codec.pth"\n')
        f.write("# The configuration name for the vocoder/decoder model.\n")
        f.write('#FISHSPEECH_DECODER_CONFIG_NAME="modded_dac_vq"\n')
        f.write("# Enable PyTorch 2.0+ model compilation for the T2S model. Set to `true` to enable.\n")
        f.write("FISHSPEECH_COMPILE=False\n")
        f.write("# Maximum number of cached reference voices (prompt tokens) to keep in memory.\n")
        f.write("#FISHSPEECH_VOICE_CACHE_SIZE=10\n")
        f.write("# Offload the Text-to-Semantic (LLaMA-based) model to the CPU when not in use to save VRAM. Set to False if you have abundant VRAM.\n")
        f.write("#FISHSPEECH_OFFLOAD_T2S_MODEL=False\n")
        f.write("# Offload the Vocoder/Decoder (DAC) model to the CPU when not in use to save VRAM. Set to False if you have abundant VRAM.\n")
        f.write("#FISHSPEECH_OFFLOAD_DECODER_MODEL=False\n")
        f.write("# The mini-batch size for the Text-to-Semantic (LLaMA) model.\n")
        f.write("# NOTE: The current engine implementation processes texts sequentially for this model, but this setting is provided for future-proofing.\n")
        f.write("#FISHSPEECH_T2S_BATCH_SIZE=1\n")
        f.write("# The mini-batch size for the vocoder/decoder model.\n")
        f.write("# Higher values can improve throughput on high-VRAM GPUs but may increase latency.\n")
        f.write("#FISHSPEECH_DECODER_BATCH_SIZE=1\n\n")

    # --- Step 9: Pre-download required models ---
    print("9. Pre-downloading required models...")
    if not engine_info["downloads"]:
        print("   - No download jobs specified for this engine. Skipping.")
    elif snapshot_download is None and any(d.get("type") == "huggingface" for d in engine_info["downloads"]):
        print("   - ERROR: huggingface-hub is not installed, but is required to download models. Skipping.")
    else:
        for job in engine_info["downloads"]:
            job_type = job.get("type")
            print(f"   - Starting download job ({job.get('description', 'No description')})")
            if job_type == "huggingface":
                repo_id = job.get("repo_id")
                local_dir = job.get("local_dir")
                if not repo_id or not local_dir:
                    print(f"     - ERROR: Invalid Hugging Face download job. Missing 'repo_id' or 'local_dir'.")
                    continue
                
                if job.get("requires_login"):
                    print(f"     - INFO: This model requires a Hugging Face account and login.")
                    print(f"     - Make sure you have accepted the model's terms on its page: https://huggingface.co/{repo_id}")

                print(f"     - Downloading '{repo_id}' to './{local_dir}'...")
                os.makedirs(Path(local_dir).parent, exist_ok=True)
                
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_dir,
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if "gated" in error_str or "restricted" in error_str or "access is denied" in error_str:
                        print("\n--- HUGGING FACE AUTHENTICATION ERROR ---")
                        print(f"Failed to download '{repo_id}' due to an access restriction.")
                        print("This model requires you to be logged into your Hugging Face account.\n")
                        print("Please follow these steps:")
                        print(f"1. Go to the model's page on Hugging Face: https://huggingface.co/{repo_id}")
                        print("2. Make sure you are logged in and have accepted the terms/gate for the model.")
                        print("3. In your terminal, run the following command and enter your access token:")
                        print("   huggingface-cli login")
                        print("   (You can get an access token from https://huggingface.co/settings/tokens)\n")
                        print("4. After logging in, re-run this configuration script:")
                        print(f"   python configure.py --engine {engine_name}")
                    else:
                        print(f"\n--- ERROR ---")
                        print(f"Failed to download from Hugging Face repo '{repo_id}'.")
                        print(f"Error details: {e}")
                    sys.exit(1)
            else:
                print(f"     - ERROR: Unknown download type '{job_type}'. Skipping.")

    # --- Final Instructions ---
    print("\n--- Configuration Complete! ---")
    if not is_conda:
        activation_cmd = f"source {venv_dir}/bin/activate" if os.name != 'nt' else f".\\{venv_dir}\\Scripts\\activate"
        print(f"1. Activate the virtual environment: {activation_cmd}")
        print("2. Run the API server: uvicorn main:app --port 8000")
    else:
        print("Your Conda environment is set up. You can now run the API server:")
        print("  uvicorn main:app --port 8000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure the TTS API for a specific engine.")
    parser.add_argument("--engine", required=True, help=f"The TTS engine to configure. Choices: {list(ENGINE_REGISTRY.keys())}")
    parser.add_argument("--cuda-version", help="Manually specify the CUDA version (e.g., '11.8'). Overrides auto-detection.")
    args = parser.parse_args()
    main(args.engine, args.cuda_version)