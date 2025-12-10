"""
Configuration file for model settings.
You can set your settings here or use environment variables.
"""

import os

# Model Configuration
# Available models: TinyLlama/TinyLlama-1.1B-Chat-v1.0
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Device Configuration
# Options: "cuda", "cpu", or "auto" (auto-detect)
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")

# Quantization Configuration (for CPU efficiency)
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "8"))  # 8 or 4

# Steering Vector Configuration
STEERING_VECTOR_STRENGTH = float(os.getenv("STEERING_VECTOR_STRENGTH", "0.5"))  # 0.0 to 1.0
STEERING_VECTOR_LAYER = int(os.getenv("STEERING_VECTOR_LAYER", "-1"))  # -1 means auto-select mid-layer

# Hugging Face Token (optional, not required for TinyLlama)
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))

