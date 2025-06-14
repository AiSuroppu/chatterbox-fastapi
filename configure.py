import os
import subprocess
import sys
import argparse
import shutil
import re

# A registry of known TTS engines and their git repositories
ENGINE_REGISTRY = {
    "chatterbox": {
        "repo": "https://github.com/resemble-ai/chatterbox.git",
        "torch_version": "2.6.0",
        "torchaudio_version": "2.6.0",
    }
}

# List of scripts to run for pre-downloading models or data
DOWNLOAD_SCRIPTS = []

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

    # --- Step 1: Prepare Python environment ---
    print("1. Preparing Python environment...")
    is_conda = 'CONDA_PREFIX' in os.environ
    venv_dir = ".venv"
    if is_conda:
        print(f"   - Detected active Conda environment: {os.environ.get('CONDA_DEFAULT_ENV')}")
        python_executable = f'"{sys.executable}"'
        pip_executable = f'"{sys.executable}" -m pip'
    else:
        print("   - Creating local virtual environment in './.venv'.")
        if not os.path.exists(venv_dir):
            run_command(f'"{sys.executable}" -m venv {venv_dir}', "Failed to create venv.")
        python_path = os.path.join(venv_dir, 'Scripts' if os.name == 'nt' else 'bin', 'python')
        python_executable = f'"{python_path}"'
        pip_path = os.path.join(venv_dir, 'Scripts' if os.name == 'nt' else 'bin', 'pip')
        pip_executable = f'"{pip_path}"'

    # --- Step 2: Clean up old vendor directory ---
    print("2. Cleaning up old vendor directory (if any)...")
    if os.path.isdir("vendor"):
        shutil.rmtree("vendor")
    
    # --- Step 3: Install PyTorch with correct CUDA support ---
    print("3. Installing PyTorch...")
    engine_info = ENGINE_REGISTRY[engine_name]
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
    # This command reads pyproject.toml and installs the base dependencies.
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
    
    # --- Step 7: Create .env file ---
    print("7. Creating .env file...")
    with open(".env", "w") as f:
        f.write(f'ENABLED_MODELS=\'["{engine_name}"]\'\n')
        f.write(f'DEVICE="{"cuda" if cuda_version else "cpu"}"\n\n')

        f.write("# NeMo text normalizer will cache compiled grammars here.\n")
        nemo_cache_path = os.path.join(os.getcwd(), ".cache", "nemo").replace("\\", "/")
        f.write(f'#NEMO_CACHE_DIR="{nemo_cache_path}"\n\n')

        f.write("# LRU cache sizes for language-specific normalizers and segmenters.\n")
        f.write("# Increase if you support many languages concurrently.\n")
        f.write("#NEMO_NORMALIZER_CACHE_SIZE=4\n")
        f.write("#PYSBD_CACHE_SIZE=4\n\n")
        
        f.write("# --- Chatterbox Specific Settings ---\n")
        f.write("# The maximum number of text segments to process in a single batch for Chatterbox.\n")
        f.write("# Increase on high-VRAM GPUs for better throughput.\n")
        f.write("#CHATTERBOX_MAX_BATCH_SIZE=2\n")
        f.write("# PyTorch 2.0+ model compilation mode for Chatterbox (e.g., 'default', 'reduce-overhead', 'max-autotune').\n")
        f.write("# Can significantly speed up inference after a one-time warm-up cost.\n")
        f.write("# Set to an empty string to disable compilation.\n")
        f.write('CHATTERBOX_COMPILE_MODE=""\n\n')
        f.write("# The maximum number of voice embeddings to keep in the server-side cache.\n")
        f.write('#CHATTERBOX_VOICE_CACHE_SIZE=50\n\n')

    # --- Step 8: Pre-download required models ---
    print("8. Pre-downloading required models...")
    if not DOWNLOAD_SCRIPTS:
        print("   - No download scripts specified. Skipping.")
    else:
        for script_path in DOWNLOAD_SCRIPTS:
            print(f"   - Running download script: {script_path}")
            run_command(f"{python_executable} {script_path}", f"Failed to run download script {script_path}.")

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