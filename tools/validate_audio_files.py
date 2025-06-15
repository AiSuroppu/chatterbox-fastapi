"""
A versatile tool to validate audio files against their original text.

This script can operate in two modes:

1.  Directory Scan Mode:
    Given a directory, it automatically finds audio files (wav, mp3, flac)
    and looks for a corresponding .txt file with the same name.
    Example: validates 'my_audio.wav' using text from 'my_audio.txt'.

2.  JSON Superblock Mode:
    If a --json-file argument is provided, it reads a JSON file (like
    'tts_state.json') to find audio file paths and their associated
    text from 'superblock' entries.

The script runs the full validation pipeline from the tts_api project on
each discovered audio-text pair and produces a detailed report, followed by
an overall statistical summary.

Usage:
  # Directory Scan Mode
  python tools/validate_audio_files.py /path/to/your/audio_and_txt_files

  # JSON Superblock Mode
  python tools/validate_audio_files.py /path/to/project --json-file tts_state.json
"""
import re
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pyphen
import torch
import torchaudio

# The script's parent is 'tools', and the parent of 'tools' is the project root.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from tts_api.core.models import PostProcessingOptions, ValidationOptions, ExportFormat
    from tts_api.services.validation import (
        ValidationContext, run_validation_pipeline, to_dbfs,
        _calculate_spectral_centroid_std_dev, SpectralCentroidValidator
    )
except ImportError as e:
    print(f"Error: Failed to import project modules. Make sure this script is in a 'tools' directory and 'src' is at the project root.")
    print(f"Import Error: {e}")
    sys.exit(1)

# Configure logging to suppress verbose debug messages from imported modules
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class TermColors:
    """Terminal colors for nice formatting."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def strip_ansi(text: str) -> str:
    """Removes ANSI escape codes from a string."""
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def discover_audio_sources(directory: Path, json_filename: Optional[str]) -> List[Tuple[Path, str]]:
    """
    Discovers audio files and their corresponding text from a given directory.

    It operates in two modes:
    1. If `json_filename` is provided, it parses that JSON file for "superblocks".
    2. Otherwise, it scans the directory for audio files and matching .txt files.

    Args:
        directory (Path): The base directory to search in.
        json_filename (Optional[str]): The name of the JSON file to parse.

    Returns:
        A list of tuples, where each tuple contains (audio_file_path, text_content).
    
    Raises:
        FileNotFoundError: If the specified JSON file is not found.
    """
    audio_text_pairs = []

    if json_filename:
        # --- JSON Superblock Mode ---
        json_path = directory / json_filename
        if not json_path.is_file():
            raise FileNotFoundError(f"Specified JSON file not found: {json_path}")

        print(f"Mode: Reading data from '{json_path.name}'...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        superblocks = data.get("superblocks", {})
        for sb_id, sb_data in superblocks.items():
            rel_audio_path = sb_data.get("active_audio_file_path")
            text = sb_data.get("concatenated_text_for_tts")
            if rel_audio_path and text:
                audio_path = (directory / rel_audio_path).resolve()
                audio_text_pairs.append((audio_path, text))
    else:
        # --- Directory Scan Mode ---
        print(f"Mode: Scanning directory for audio and .txt files...")
        audio_files: Dict[str, Path] = {}
        supported_extensions = [f".{fmt.value}" for fmt in ExportFormat]

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                audio_files[file_path.stem] = file_path

        for stem, audio_path in audio_files.items():
            txt_path = directory / f"{stem}.txt"
            if txt_path.is_file():
                try:
                    text = txt_path.read_text(encoding='utf-8').strip()
                    if text:
                        audio_text_pairs.append((audio_path, text))
                except Exception as e:
                    print(f"{TermColors.WARNING}Warning: Could not read text file '{txt_path}': {e}{TermColors.ENDC}")

    return audio_text_pairs


def format_result(is_ok: bool, reason: str = "") -> str:
    """Formats a boolean result with color and an optional reason."""
    if is_ok:
        return f"{TermColors.OKGREEN}PASS{TermColors.ENDC}"
    else:
        return f"{TermColors.FAIL}FAIL{TermColors.ENDC} ({reason})"


def print_validation_report(context: ValidationContext) -> dict:
    """
    Calculates and prints a detailed, step-by-step validation report
    for a given audio file and its context in a compact table format.

    Returns:
        dict: A dictionary containing the raw numeric values calculated,
              to be used for aggregate statistics.
    """
    params = context.validation_params
    report_values = {
        "Signal Level (dBFS)": None,
        "Voiced Segments Found": None,
        "Segments Removed": None,
        "Voiced Duration / Syllable (s)": None,
        "Max Contiguous Silence (s)": None,
        "Clipping (Voiced) (%)": None,
        "Spectral Centroid Std Dev (Hz)": None,
    }

    def get_status_str(is_ok: bool, is_na: bool = False, is_info: bool = False):
        if is_na: return f"{TermColors.OKCYAN}N/A{TermColors.ENDC}"
        if is_info: return f"{TermColors.OKBLUE}INFO{TermColors.ENDC}"
        if is_ok: return f"{TermColors.OKGREEN}PASS{TermColors.ENDC}"
        return f"{TermColors.FAIL}FAIL{TermColors.ENDC}"

    # --- 1. Collect data for all rows ---
    table_data = []

    # Signal Level
    dbfs = to_dbfs(context.original_waveform)
    report_values["Signal Level (dBFS)"] = dbfs
    is_ok = params.max_silence_dbfs is None or dbfs >= params.max_silence_dbfs
    details = "Disabled" if params.max_silence_dbfs is None else f"> {params.max_silence_dbfs} dBFS"
    table_data.append(["Signal Level", f"{dbfs:.1f} dBFS", get_status_str(is_ok, is_na=(params.max_silence_dbfs is None)), details])

    # VAD & Segment Filtering
    initial_segs = context.initial_speech_segments
    filtered_segs = context.filtered_speech_segments
    removed_count = len(initial_segs) - len(filtered_segs)
    report_values["Voiced Segments Found"] = len(filtered_segs)
    report_values["Segments Removed"] = removed_count
    table_data.append(["Voiced Segments Found", f"{len(filtered_segs)} / {len(initial_segs)}", get_status_str(True, is_info=True), f"{removed_count} low-energy segment(s) removed"])
    
    if not filtered_segs:
        # Add placeholders if no speech is found
        for check_name in ["Voiced Duration / Syllable", "Max Contiguous Silence", "Clipping (Voiced)", "Spectral Centroid Std Dev"]:
            table_data.append([check_name, "---", get_status_str(True, is_na=True), "No voiced segments"])
    else:
        # Voiced Duration
        try:
            dic = pyphen.Pyphen(lang=context.language)
            syllable_count = sum(len(dic.inserted(word).split('-')) for word in context.text_chunk.split())
            if syllable_count == 0: syllable_count = 1
            total_voiced_samples = sum(s['end'] - s['start'] for s in filtered_segs)
            total_voiced_seconds = total_voiced_samples / context.sample_rate
            ratio = total_voiced_seconds / syllable_count if syllable_count > 0 else 0
            report_values["Voiced Duration / Syllable (s)"] = ratio
            is_ok_min = params.min_voiced_duration_per_syllable is None or ratio >= params.min_voiced_duration_per_syllable
            is_ok_max = params.max_voiced_duration_per_syllable is None or ratio <= params.max_voiced_duration_per_syllable
            is_na = params.min_voiced_duration_per_syllable is None and params.max_voiced_duration_per_syllable is None
            details = "Disabled" if is_na else f"in [{params.min_voiced_duration_per_syllable}, {params.max_voiced_duration_per_syllable}]"
            table_data.append(["Voiced Duration / Syllable", f"{ratio:.3f} s", get_status_str(is_ok_min and is_ok_max, is_na=is_na), details])
        except Exception as e:
            table_data.append(["Voiced Duration / Syllable", "Error", get_status_str(False), str(e)])

        # Max Contiguous Silence
        last_speech_end = 0
        max_silence_samples = 0
        for seg in filtered_segs:
            max_silence_samples = max(max_silence_samples, seg['start'] - last_speech_end)
            last_speech_end = seg['end']
        max_s = max_silence_samples / context.sample_rate
        report_values["Max Contiguous Silence (s)"] = max_s
        is_ok = params.max_contiguous_silence_s is None or max_s <= params.max_contiguous_silence_s
        details = "Disabled" if params.max_contiguous_silence_s is None else f"< {params.max_contiguous_silence_s} s"
        table_data.append(["Max Contiguous Silence", f"{max_s:.2f} s", get_status_str(is_ok, is_na=(params.max_contiguous_silence_s is None)), details])

        # Clipping
        total_speech_samples = sum(s['end'] - s['start'] for s in filtered_segs)
        total_clipped_samples = sum(torch.sum((context.original_waveform[..., s['start']:s['end']].abs() >= 0.999)).item() for s in filtered_segs)
        clip_percent = (total_clipped_samples / total_speech_samples) * 100 if total_speech_samples > 0 else 0.0
        report_values["Clipping (Voiced) (%)"] = clip_percent
        is_ok = params.max_clipping_percentage is None or clip_percent <= params.max_clipping_percentage
        details = "Disabled" if params.max_clipping_percentage is None else f"< {params.max_clipping_percentage:.1f}%"
        table_data.append(["Clipping (Voiced)", f"{clip_percent:.3f}%", get_status_str(is_ok, is_na=(params.max_clipping_percentage is None)), details])

        # Spectral Centroid Std Dev
        val_str = "---"
        is_na_spec = params.min_spectral_centroid_std_dev is None
        details = "Disabled" if is_na_spec else f"> {params.min_spectral_centroid_std_dev} Hz"

        # Get the official pass/fail status from the validator
        spec_result = SpectralCentroidValidator().is_valid(context)
        is_ok = spec_result.is_ok
        if not is_na_spec:
            calculated_std_dev = _calculate_spectral_centroid_std_dev(context)
            if calculated_std_dev is not None:
                val_str = f"{calculated_std_dev:.1f} Hz"
                report_values["Spectral Centroid Std Dev (Hz)"] = calculated_std_dev
            else:
                details = "Needs >= 2 voiced segments"
        table_data.append(["Spectral Centroid Std Dev", val_str, get_status_str(is_ok, is_na=is_na_spec), details])

    # --- 2. Print the formatted table ---
    headers = ["Check", "Value", "Status", "Details / Threshold"]
    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(strip_ansi(str(cell))))

    def print_row(row_items):
        cells = []
        for i, item in enumerate(row_items):
            stripped_item = strip_ansi(str(item))
            padding = " " * (col_widths[i] - len(stripped_item))
            cells.append(f" {item}{padding} ")
        print(f"|{'|'.join(cells)}|")
    line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(f"\n{TermColors.BOLD}--- Validation Report ---{TermColors.ENDC}")
    print(line)
    print_row(headers)
    print(line.replace('-', '='))
    for row in table_data:
        print_row(row)
    print(line)
    return report_values


def print_summary_report(stats_data: dict, total_files: int):
    """
    Calculates and prints a summary report with statistics for all processed files.

    Args:
        stats_data (dict): A dictionary where keys are parameter names and
                           values are lists of collected numeric values.
        total_files (int): The total number of files successfully processed.
    """
    summary_header = " Overall Summary Report "
    print(f"\n{TermColors.HEADER}{summary_header.center(80, '=')}{TermColors.ENDC}")
    if total_files == 0:
        print("\nNo audio files were successfully processed. No summary to display.\n")
        return
    print(f"\nStatistics calculated from {TermColors.BOLD}{total_files}{TermColors.ENDC} successfully processed audio file(s).\n")
    headers = ["Validation Parameter", "Count", "Mean", "Std Dev", "Min", "Max"]
    col_widths = [len(h) for h in headers]
    table_data = []
    for param, values in stats_data.items():
        if not values: row = [param, "0", "N/A", "N/A", "N/A", "N/A"]
        else:
            count = len(values)
            # Use torch for stats to avoid new dependencies
            tensor_values = torch.tensor(values, dtype=torch.float32)
            mean = torch.mean(tensor_values).item()
            # std() needs at least 2 elements
            std_dev = torch.std(tensor_values).item() if count > 1 else 0.0
            min_val = torch.min(tensor_values).item()
            max_val = torch.max(tensor_values).item()

            # For integer-based stats, format without decimals for min/max
            if "Segments" in param:
                 row = [
                    param,
                    f"{count}",
                    f"{mean:.2f}",
                    f"{std_dev:.2f}",
                    f"{int(min_val)}",
                    f"{int(max_val)}",
                ]
            else:
                row = [
                    param,
                    f"{count}",
                    f"{mean:.3f}",
                    f"{std_dev:.3f}",
                    f"{min_val:.3f}",
                    f"{max_val:.3f}",
                ]
        table_data.append(row)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Printing logic adapted from print_validation_report
    def print_row(row_items):
        cells = []
        for i, item in enumerate(row_items):
            padding = " " * (col_widths[i] - len(str(item)))
            cells.append(f" {item}{padding} ")
        print(f"|{'|'.join(cells)}|")
    line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(line)
    print_row([h for h in headers])
    print(line.replace('-', '='))
    for row in table_data:
        print_row(row)
    print(line)


def main():
    """Main function to parse arguments and run the validation."""
    parser = argparse.ArgumentParser(
        description="Validate audio files using text from a JSON file or corresponding .txt files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Scan a directory for matching audio and .txt files
  python tools/validate_audio_files.py /path/to/my_dataset

  # Use a JSON file to define audio/text pairs
  python tools/validate_audio_files.py /path/to/project/output --json-file tts_state.json
"""
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="The directory containing audio files and text sources."
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Optional: Name of a JSON file in the directory to read 'superblocks' from."
    )
    args = parser.parse_args()

    data_directory = args.directory.resolve()
    if not data_directory.is_dir():
        print(f"{TermColors.FAIL}Error: Directory not found at '{data_directory}'{TermColors.ENDC}")
        sys.exit(1)

    try:
        audio_text_pairs = discover_audio_sources(data_directory, args.json_file)
    except FileNotFoundError as e:
        print(f"{TermColors.FAIL}Error: {e}{TermColors.ENDC}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{TermColors.FAIL}Error: Failed to parse JSON file '{args.json_file}': {e}{TermColors.ENDC}")
        sys.exit(1)


    if not audio_text_pairs:
        print(f"{TermColors.WARNING}No audio/text pairs found to validate.{TermColors.ENDC}")
        return

    print(f"Found {len(audio_text_pairs)} audio/text pair(s) to validate.\n")

    all_validation_values = {
        "Signal Level (dBFS)": [],
        "Voiced Segments Found": [],
        "Segments Removed": [],
        "Voiced Duration / Syllable (s)": [],
        "Max Contiguous Silence (s)": [],
        "Clipping (Voiced) (%)": [],
        "Spectral Centroid Std Dev (Hz)": []
    }
    processed_file_count = 0

    for audio_path, text in audio_text_pairs:
        rel_audio_path = audio_path.relative_to(data_directory)
        header = f" VALIDATING: {rel_audio_path} "
        print(f"{TermColors.HEADER}{'=' * len(header)}{TermColors.ENDC}")
        print(f"{TermColors.HEADER}{header}{TermColors.ENDC}")
        print(f"{TermColors.HEADER}{'=' * len(header)}{TermColors.ENDC}")

        if not audio_path.is_file():
            print(f"\n{TermColors.FAIL}Error: Audio file not found at '{audio_path}'{TermColors.ENDC}\n")
            continue

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate

            print(f"  - Audio Path: {audio_path}")
            print(f"  - Duration: {duration:.2f} s")
            print(f"  - Sample Rate: {sample_rate} Hz")
            print(f"  - Text Snippet: \"{text[:70].replace(chr(10), ' ')}...\"")

            # Use default params as they are not stored in the state file
            validation_params = ValidationOptions()
            post_processing_params = PostProcessingOptions()
            language = "en"  # Assume English for validation

            context = ValidationContext(
                original_waveform=waveform,
                sample_rate=sample_rate,
                text_chunk=text,
                validation_params=validation_params,
                post_processing_params=post_processing_params,
                language=language,
            )

            report_values = print_validation_report(context)

            for key, value in report_values.items():
                if value is not None:
                    all_validation_values[key].append(value)
            processed_file_count += 1

            final_result = run_validation_pipeline(
                waveform=waveform,
                sample_rate=sample_rate,
                text_chunk=text,
                validation_params=validation_params,
                post_processing_params=post_processing_params,
                language=language
            )

            print(f"\n{'-' * (len(header))}")
            print(f"{TermColors.BOLD}OVERALL RESULT: {format_result(final_result.is_ok, final_result.reason)}{TermColors.ENDC}")
            print(f"{'-' * (len(header))}\n")

        except Exception as e:
            print(f"\n{TermColors.FAIL}An unexpected error occurred while processing '{rel_audio_path}':{TermColors.ENDC}")
            print(f"  {e}")
            print("\n")

    print_summary_report(all_validation_values, processed_file_count)


if __name__ == "__main__":
    main()