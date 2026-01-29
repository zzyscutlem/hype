"""Setup script for HyPE package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hype-agent",
    version="0.1.0",
    author="HyPE Team",
    description="Hypothesis-Driven Planning and Semantic Adaptation with Evolutionary Principle-Value Distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",  # For LoRA
        "pymilvus>=2.3.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "hypothesis>=6.80.0",  # For property-based testing
        "pytest>=7.4.0",
        "accelerate>=0.20.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",  # For experiment tracking
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest-cov>=4.1.0",
            "ipython>=8.14.0",
        ],
    },
)
