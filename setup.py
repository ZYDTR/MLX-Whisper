"""安装脚本"""
from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlx-speech-pipeline",
    version="0.1.0",
    description="Apple Silicon 本地语音处理流水线",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.10.0",
        "lightning-whisper-mlx>=0.2.0",
        "senko>=0.1.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "numpy>=1.24.0",
        "onnxruntime>=1.16.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "speech-pipeline=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)

