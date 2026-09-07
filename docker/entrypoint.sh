#!/bin/bash
set -e

echo "[BOOT] Initializing PAR Inference Container..."

echo "[BOOT] Initializing PAR Inference Container..."
if [ ! -f "inference/models/baseline_v4_prod.onnx" ]; then
    echo "[ERROR] baseline_v4_prod.onnx is missing from container working directory!"
    exit 1
fi

echo "[BOOT] Waiting for video stream source to become available..."
sleep 3

echo "[BOOT] Environment checks passed. Launching inference engine..."
exec python3 inference/inference.py "$@"