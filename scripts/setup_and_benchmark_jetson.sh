#!/usr/bin/env bash
# ==============================================================================
# Automated Setup & Benchmark Script for NVIDIA Jetson Orin Nano / Nano DevKit
# Course: Foundations of Data Science (FDS) — November 2026
# ==============================================================================
set -e

echo "===================================================================="
echo "  Constrained Graph Kinematics: Jetson Hardware Benchmark Setup"
echo "===================================================================="

# 1. Maximize compute performance clocks
echo "[1/4] Configuring Jetson MAXN performance profile..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0 || true
fi
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks || true
fi

# 2. Install lightweight edge runtime dependencies
echo "[2/4] Installing Python dependencies (onnxruntime, numpy)..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-numpy

# Install onnxruntime via pip (or apt if available)
pip3 install --quiet onnxruntime || pip install --quiet onnxruntime

# 3. Verify ONNX model existence
MODEL_PATH="checkpoints/student_kinematic_fk.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[-] Error: Model $MODEL_PATH not found. Ensure you are in the repository root."
    exit 1
fi
echo "[+] Detected model: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"

# 4. Run the physical edge benchmark
echo "[3/4] Running physical edge benchmark across 100 frames..."
python3 deployment/benchmark_edge.py --output results/benchmark_jetson_nano.json

echo "===================================================================="
echo "  [SUCCESS] Benchmark Complete!"
echo "  Results written to: results/benchmark_jetson_nano.json"
echo "===================================================================="
cat results/benchmark_jetson_nano.json
