# 🎙️ Kannada Text-to-Speech (TTS) System

A comprehensive Kannada TTS system with CPU optimization and Mac-specific performance enhancements. This project provides multiple approaches to synthesize high-quality Kannada speech from text.

## 🚀 Features

- **CPU-Only Operation**: Works without GPU requirements
- **Mac Optimization**: Special optimizations for M1/M2 MacBooks  
- **Multi-Speaker Support**: Handles both single and multi-speaker models
- **Automatic Fallbacks**: Multiple synthesis strategies for reliability
- **Model Quantization**: Reduced memory usage and faster inference

## 📋 Prerequisites

### System Requirements
- **Python 3.7+**
- **PyTorch** (CPU version supported)
- **4GB+ RAM** recommended
- **2GB+ free disk space** for models

### Operating Systems
- ✅ **macOS** (optimized for M1/M2)
- ✅ **Linux** (Ubuntu, CentOS, etc.)
- ✅ **Windows** (10/11)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/arushi-vaidya/Indic-Speech-Translation.git
cd kannada_tts_fast
```

### 2. Install Dependencies
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TTS library
pip install TTS

# Install additional dependencies
pip install numpy pathlib
```

### 3. Download Kannada Models

You need to obtain Kannada TTS models and organize them as follows:
```bash
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/kn.zip
unzip kn.zip
```
copy this to **models/v1/kn** as well
```
kannada_tts_fast/
├── kn/
│   ├── fastpitch/
│   │   ├── best_model.pth      # FastPitch TTS model
│   │   ├── config.json         # FastPitch configuration
│   │   └── speakers.pth        # Speaker embeddings (optional)
│   └── hifigan/
│       ├── best_model.pth      # HiFi-GAN vocoder model
│       └── config.json         # HiFi-GAN configuration
├── models/v1/kn
│   ├── fastpitch/
│   │   ├── best_model.pth      # FastPitch TTS model
│   │   ├── config.json         # FastPitch configuration
│   │   └── speakers.pth        # Speaker embeddings (optional)
│   └── hifigan/
│       ├── best_model.pth      # HiFi-GAN vocoder model
│       └── config.json         # HiFi-GAN configuration
├── kannada_tts.py
├── mac_optimize.py
└── README.md
```

### Instructions

```bash
python3 mac_optimize.py
python3 mac_tts.py "ನಮಸ್ಕಾರ" optimized_output.wav
```

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `kannada_tts.py` | Main TTS script with CUDA patches for CPU-only operation |
| `mac_optimize.py` | Mac-specific model optimization and quantization |
| `mac_tts.py` | Optimized TTS runner for Mac (generated after optimization) |
| `fix_speakers.py` | Utility to create missing speakers.pth file |
| `kannada_tts_no_speakers.py` | Simplified TTS that works without speakers.pth |


## Required files:
1. ✅ kn/fastpitch/best_model.pth
2. ✅ kn/fastpitch/config.json  
3. ✅ kn/hifigan/best_model.pth
4. ✅ kn/hifigan/config.json
5. ⚠️ kn/fastpitch/speakers.pth (optional, can be generated)
```
---