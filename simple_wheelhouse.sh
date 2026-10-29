#!/bin/bash
# Simple wheelhouse builder - just download vllm and all dependencies

OUTPUT_DIR="vllm_wheelhouse_v0271_cu129"
mkdir -p "$OUTPUT_DIR"

echo "=== Downloading vLLM 0.27.1 and all dependencies ==="
python -m pip download vllm==0.27.1 \
    --extra-index-url https://flashinfer.ai/whl/cu129 \
    -d "$OUTPUT_DIR"

echo "=== Creating requirements.lock ==="
cd "$OUTPUT_DIR"
python -m pip freeze > requirements.lock
cd ..

echo "=== Wheelhouse complete ==="
echo "Total wheels: $(ls -1 "$OUTPUT_DIR"/*.whl 2>/dev/null | wc -l)"
echo "Upload $OUTPUT_DIR to Kaggle"
