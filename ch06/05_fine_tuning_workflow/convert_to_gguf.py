"""
Convert a Hugging Face model to GGUF format for use with llama.cpp.
This script assumes you have llama.cpp installed and the conversion script available.
"""

import os
import subprocess
import sys
from pathlib import Path


def convert_to_gguf(model_path, output_path=None, outtype="f16"):
    """
    Convert a Hugging Face model to GGUF format.

    Args:
        model_path (str): Path to the Hugging Face model directory.
        output_path (str, optional): Path for the output GGUF file. If None, uses model_dir/model.gguf.
        outtype (str): Output type for the GGUF file (e.g., "f16", "q4_0", "q5_1"). Default is "f16".
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    if output_path is None:
        output_path = model_path / f"model.{outtype}.gguf"
    else:
        output_path = Path(output_path)

    # Path to the llama.cpp conversion script
    # Assume llama.cpp is cloned in the same directory as this script or in a known location
    # We'll look for it in the parent directory or use an environment variable
    llama_cpp_dir = os.getenv("LLAMA_CPP_DIR", Path(__file__).parent / "llama.cpp")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"Error: Conversion script not found at {convert_script}")
        print("Please clone llama.cpp and set the LLAMA_CPP_DIR environment variable.")
        print("Example: git clone https://github.com/ggerganov/llama.cpp")
        print("Then set LLAMA_CPP_DIR to the path of the cloned repository.")
        return False

    # Run the conversion script
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype,
    ]

    print(f"Running conversion command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Conversion successful!")
        print(f"GGUF model saved to: {output_path}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face model to GGUF format."
    )
    parser.add_argument("model_path", help="Path to the Hugging Face model directory.")
    parser.add_argument("--output", help="Path for the output GGUF file.", default=None)
    parser.add_argument(
        "--outtype", help="Output type for GGUF (e.g., f16, q4_0, q5_1).", default="f16"
    )

    args = parser.parse_args()

    success = convert_to_gguf(args.model_path, args.output, args.outtype)
    sys.exit(0 if success else 1)
