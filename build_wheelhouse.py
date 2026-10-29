#!/usr/bin/env python3
"""
Build a complete vLLM wheelhouse for Kaggle ARC-AGI-3 competition.

This script downloads all required wheels for vllm, torch, flashinfer and their dependencies,
then packages them for upload to Kaggle.

Usage:
    python build_wheelhouse.py --output-dir ./wheelhouse_output
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
    return result


def get_python_version() -> str:
    """Get the current Python version (e.g., '3.10')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def download_wheels(
    output_dir: Path,
    python_version: str,
    cuda_version: str = "cu129",
    torch_version: str = "2.5.1",
    vllm_version: str = "0.27.1",
    flashinfer_version: str = "0.1.6",
) -> None:
    """
    Download wheels for vLLM, torch, flashinfer and all dependencies.

    Args:
        output_dir: Directory to save wheels
        python_version: Python version (e.g., '3.10')
        cuda_version: CUDA version (e.g., 'cu129' for CUDA 12.9)
        torch_version: PyTorch version
        vllm_version: vLLM version
        flashinfer_version: flashinfer version
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download PyTorch wheels first
    print(f"\n=== Downloading PyTorch {torch_version} ===")
    torch_cmd = [
        sys.executable, "-m", "pip", "download",
        f"torch=={torch_version}",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--no-deps",
        "-d", str(output_dir),
    ]
    run_command(torch_cmd)

    # Download vLLM and all dependencies (includes flashinfer)
    print(f"\n=== Downloading vLLM {vllm_version} and dependencies ===")
    vllm_cmd = [
        sys.executable, "-m", "pip", "download",
        f"vllm=={vllm_version}",
        "--extra-index-url", f"https://flashinfer.ai/whl/{cuda_version}",
        "-d", str(output_dir),
    ]
    run_command(vllm_cmd)

    # Download additional commonly missing packages
    print(f"\n=== Downloading additional packages ===")
    additional_packages = [
        "aiohttp>=3.9.0",
        "blake3>=1.0.0",
        "aiosignal>=1.0.0",
        "aiohappyeyeballs>=2.0.0",
    ]
    for pkg in additional_packages:
        cmd = [
            sys.executable, "-m", "pip", "download",
            pkg,
            "-d", str(output_dir),
        ]
        run_command(cmd)


def create_requirements_lock(
    output_dir: Path,
    torch_version: str = "2.5.1",
    vllm_version: str = "0.27.1",
    flashinfer_version: str = "0.1.6",
) -> None:
    """Create a requirements.lock file with exact versions."""
    # Get all wheel files
    wheel_files = list(output_dir.glob("*.whl"))
    
    # Extract package versions from wheel filenames
    package_versions = {}
    for wheel in wheel_files:
        # Format: package_name-version-py_tag-abi_tag-platform_tag.whl
        parts = wheel.stem.split('-')
        if len(parts) >= 2:
            pkg_name = parts[0].replace('_', '-')
            pkg_version = parts[1]
            package_versions[pkg_name] = pkg_version
    
    # Add key packages explicitly
    package_versions["torch"] = torch_version
    package_versions["vllm"] = vllm_version
    package_versions["flashinfer"] = flashinfer_version
    
    # Write requirements.lock
    requirements_file = output_dir / "requirements.lock"
    with open(requirements_file, "w") as f:
        for pkg, version in sorted(package_versions.items()):
            f.write(f"{pkg}=={version}\n")
    
    print(f"\n=== Created requirements.lock with {len(package_versions)} packages ===")
    print(f"Location: {requirements_file}")


def create_metadata(output_dir: Path) -> None:
    """Create metadata file with build information."""
    metadata = {
        "wheelhouse_info": {
            "python_version": get_python_version(),
            "torch_version": "2.5.1",
            "vllm_version": "0.27.1",
            "flashinfer_version": "0.1.6",
            "cuda_version": "cu129",
            "build_date": subprocess.run(
                ["date", "-Iseconds"],
                capture_output=True,
                text=True
            ).stdout.strip(),
        }
    }
    
    metadata_file = output_dir / "wheelhouse_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== Created wheelhouse metadata ===")
    print(f"Location: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Build vLLM wheelhouse for Kaggle")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./vllm_wheelhouse_v0271_cu129"),
        help="Output directory for wheels",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default=None,
        help="Python version (default: auto-detect)",
    )
    parser.add_argument(
        "--cuda-version",
        type=str,
        default="cu129",
        help="CUDA version (default: cu129)",
    )
    parser.add_argument(
        "--torch-version",
        type=str,
        default="2.5.1",
        help="PyTorch version (default: 2.5.1)",
    )
    parser.add_argument(
        "--vllm-version",
        type=str,
        default="0.27.1",
        help="vLLM version (default: 0.27.1)",
    )
    parser.add_argument(
        "--flashinfer-version",
        type=str,
        default="0.1.6",
        help="flashinfer version (default: 0.1.6)",
    )
    
    args = parser.parse_args()
    
    python_version = args.python_version or get_python_version()
    
    print(f"=== Building vLLM Wheelhouse ===")
    print(f"Python version: {python_version}")
    print(f"CUDA version: {args.cuda_version}")
    print(f"PyTorch version: {args.torch_version}")
    print(f"vLLM version: {args.vllm_version}")
    print(f"flashinfer version: {args.flashinfer_version}")
    print(f"Output directory: {args.output_dir}")
    
    # Download wheels
    download_wheels(
        output_dir=args.output_dir,
        python_version=python_version,
        cuda_version=args.cuda_version,
        torch_version=args.torch_version,
        vllm_version=args.vllm_version,
        flashinfer_version=args.flashinfer_version,
    )
    
    # Create requirements.lock
    create_requirements_lock(
        output_dir=args.output_dir,
        torch_version=args.torch_version,
        vllm_version=args.vllm_version,
        flashinfer_version=args.flashinfer_version,
    )
    
    # Create metadata
    create_metadata(args.output_dir)
    
    print(f"\n=== Wheelhouse build complete ===")
    print(f"Total wheels: {len(list(args.output_dir.glob('*.whl')))}")
    print(f"Upload {args.output_dir} to Kaggle as a dataset")
    print(f"\nThen update your kaggle.py to use:")
    print(f"  DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE = 'your-username/your-wheelhouse-dataset'")


if __name__ == "__main__":
    main()
