"""
Download required models for HyPE system using HuggingFace mirror.
"""

import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Set HuggingFace mirror endpoint for China region
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("Downloading models using HuggingFace mirror...")
print("=" * 60)

# Download base model (Qwen-2.5-3B-Instruct)
print("\n1. Downloading Qwen-2.5-3B-Instruct...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    print("✓ Qwen-2.5-3B-Instruct downloaded successfully")
except Exception as e:
    print(f"✗ Failed to download Qwen-2.5-3B-Instruct: {e}")

# Download embedding model (BGE-large-en-v1.5)
print("\n2. Downloading BGE-large-en-v1.5...")
try:
    embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("✓ BGE-large-en-v1.5 downloaded successfully")
except Exception as e:
    print(f"✗ Failed to download BGE-large-en-v1.5: {e}")

print("\n" + "=" * 60)
print("Model download complete!")
print("=" * 60)
