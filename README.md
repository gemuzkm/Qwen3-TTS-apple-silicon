# Qwen3-TTS for Apple Silicon

Optimized implementation of Qwen3-TTS text-to-speech model for Apple Silicon Macs (M1/M2/M3/M4). This guide provides detailed installation instructions, configuration options, and performance optimization tips for running Qwen3-TTS on macOS with Metal Performance Shaders (MPS) acceleration.

## 🎯 Features

- ✅ Native Apple Silicon (M1/M2/M3/M4) support with MPS acceleration
- ✅ Optimized `bfloat16` precision for better performance
- ✅ SDPA (Scaled Dot-Product Attention) implementation for MPS compatibility
- ✅ Gradio web interface for easy interaction
- ✅ Support for both CPU fallback and MPS acceleration
- ✅ Memory efficient (~7.2 GB with bfloat16)

## 📋 Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10 or higher
- 8GB+ RAM (16GB recommended)
- HuggingFace account and access token

## 🔑 HuggingFace Token Setup

### Why do you need a token?

Qwen3-TTS model requires authentication to download from HuggingFace. You need to create a free account and generate an access token.

### Step-by-step guide:

1. **Create a HuggingFace account**
   - Go to [https://huggingface.co/join](https://huggingface.co/join)
   - Sign up with your email or GitHub account

2. **Generate an Access Token**
   - Navigate to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Give it a name (e.g., "Qwen3-TTS")
   - Select token type: **Read** (sufficient for model downloads)
   - Click "Generate token"
   - **Important:** Copy the token immediately - you won't see it again!

3. **Configure the token**

   **Option A: Environment Variable (Recommended)**
   ```bash
   # Add to your ~/.zshrc or ~/.bash_profile
   export HF_TOKEN="hf_YourTokenHere"
   
   # Reload your shell
   source ~/.zshrc
   ```

   **Option B: In Python script**
   ```python
   # Create a file: config.py
   HF_TOKEN = "hf_YourTokenHere"
   ```
   ⚠️ **Never commit this file to Git!** (see .gitignore section below)

   **Option C: Login via CLI**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # Paste your token when prompted
   ```

4. **Accept model license**
   - Visit [Qwen3-TTS model page](https://huggingface.co/Qwen/Qwen3-TTS)
   - Click "Agree and access repository" if prompted

## 🚀 Quick Start

### Installation Method 1: Using venv (Recommended)

```bash
# 1. Install system dependencies
brew install portaudio ffmpeg sox

# 2. Create project directory
mkdir qwen3-tts-app && cd qwen3-tts-app

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install PyTorch with MPS support
pip install --upgrade pip
pip install torch torchvision torchaudio

# 5. Install dependencies
pip install \
  "transformers==4.57.3" \
  "accelerate==1.12.0" \
  qwen-tts \
  gradio \
  einops \
  librosa \
  soundfile \
  sox \
  onnxruntime \
  numpy

# 6. Clone or download app_mac_m4.py
# (Place your app_mac_m4.py file here)

# 7. Set your HuggingFace token
export HF_TOKEN="hf_YourTokenHere"

# 8. Run the application
python app_mac_m4.py
```

### Installation Method 2: Using Conda

```bash
# 1. Install system dependencies
brew install portaudio ffmpeg sox

# 2. Create conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# 3. Install all packages via pip (even with conda)
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install qwen-tts gradio transformers==4.57.3 accelerate==1.12.0

# 4. Set HuggingFace token and run
export HF_TOKEN="hf_YourTokenHere"
python app_mac_m4.py
```

### 🌐 Access the Web Interface

After running the script, open your browser and navigate to:
```
http://127.0.0.1:7860
```

## ⚙️ Configuration Options

### Precision Mode: bfloat16 vs float32

**Current default: `bfloat16` (Recommended)**

#### Performance Comparison

| Metric | bfloat16 | float32 |
|--------|----------|---------|
| **Memory Usage** | ~7.2 GB | ~8.2 GB |
| **Speed** | 100% (baseline) | 90-95% |
| **Stability** | ✅ Stable on M4 (Jan 2026) | ✅✅ Maximum |
| **Precision** | Good for TTS | Excellent |
| **Dynamic Range** | Same as float32 | Standard |
| **Recommended** | ✅ YES | Only if issues occur |

#### Why bfloat16?

- **Official recommendation**: Qwen3-TTS documentation recommends `torch.bfloat16` for Apple Silicon
- **Hardware support**: M1/M2/M3/M4 have native bfloat16 acceleration
- **Better efficiency**: 10-15% faster inference, ~1GB less memory
- **Wide dynamic range**: Maintains float32-like range despite lower precision

#### When to use float32?

Switch to float32 if you experience:
- Silent or low-quality audio output
- Generation freezes or crashes
- "BFloat16 not supported" errors
- Inconsistent results across different texts

#### How to switch precision modes:

```python
# Add environment variable support in your script:
import os
import torch

DTYPE_STR = os.environ.get('QWEN_DTYPE', 'bfloat16')
TORCH_DTYPE = torch.float32 if DTYPE_STR == 'float32' else torch.bfloat16

# Then in model loading:
model = Qwen3TTSModel.from_pretrained(
    model_path,
    torch_dtype=TORCH_DTYPE,
    attn_implementation="sdpa",
    device_map=DEVICE,
    token=HF_TOKEN,
)
```

Run with float32:
```bash
export QWEN_DTYPE="float32"
python app_mac_m4.py
```

## 📁 Project Structure & .gitignore

### Recommended .gitignore

Create a `.gitignore` file in your project root to exclude unnecessary files:

```gitignore
# Python virtual environments
venv/
env/
ENV/
.venv/

# Conda environments
conda-meta/

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Model cache and downloads
models/
*.bin
*.safetensors
checkpoints/

# HuggingFace cache
.cache/
transformers_cache/

# Sensitive information
config.py
.env
*.key
*.token

# Generated audio files
output/
*.wav
*.mp3

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Logs
*.log
logs/

# Temporary files
temp/
tmp/
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "MPS backend not available"

**Solution:**
```bash
# Check PyTorch MPS support
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# If False, reinstall PyTorch:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

#### 2. "HuggingFace Token Error"

**Symptoms:**
- `401 Unauthorized` error
- "Repository not found" error

**Solution:**
```bash
# Verify token is set:
echo $HF_TOKEN

# Login via CLI:
huggingface-cli login

# Or set in script:
from huggingface_hub import login
login(token="hf_YourTokenHere")
```

#### 3. "UserWarning: Trying to convert audio automatically"

**This is NORMAL!** Gradio automatically converts float32 audio to int16 for browser playback. It does NOT affect audio quality.

To suppress the warning:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*audio automatically.*')
```

#### 4. Slow Model Download

The model is approximately 4GB. For faster downloads in some regions:

```bash
# Use HuggingFace mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com
python app_mac_m4.py
```

#### 5. "Port 7860 already in use"

**Solution:**
```python
# Modify demo.launch() in your script:
demo.launch(
    server_name="127.0.0.1",
    server_port=7861,  # Use different port
    share=False
)
```

## 🧪 Testing Your Setup

### Verify bfloat16 support:

```python
import torch

print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

# Test bfloat16 tensor
x = torch.randn(10, dtype=torch.bfloat16, device='mps')
print("BFloat16 support:", x.dtype)  # Should print: torch.bfloat16
print("✅ BFloat16 is working!")
```

### Model loading test:

```python
from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS",
    torch_dtype=torch.bfloat16,
    device_map="mps",
    token="hf_YourTokenHere"
)
print("✅ Model loaded successfully!")
```

## 📊 Performance Benchmarks

### Apple M4 (16GB RAM):
- **Load time**: ~15-20 seconds
- **Generation speed**: ~1.5-2.0x realtime
- **Memory usage**: ~7.2 GB (bfloat16)
- **Audio quality**: Excellent

### Apple M1/M2 (8GB RAM):
- **Load time**: ~20-30 seconds
- **Generation speed**: ~1.0-1.5x realtime
- **Memory usage**: ~7.5 GB (bfloat16)
- **Note**: May experience occasional slowdowns with limited RAM

## ❓ FAQ

**Q: Why use bfloat16 instead of float32?**  
A: Apple Silicon has hardware-accelerated bfloat16 support, resulting in 10-15% faster inference and reduced memory usage while maintaining excellent audio quality.

**Q: Why SDPA instead of Flash Attention 2?**  
A: Flash Attention is not compatible with Metal Performance Shaders (MPS). SDPA (Scaled Dot-Product Attention) is PyTorch's standard attention mechanism that works across all platforms.

**Q: Do I need portaudio and sox?**  
A: Yes, these are system-level audio processing libraries required by Python audio packages (soundfile, librosa). Without them, you'll encounter import errors.

**Q: Can this run on Intel Macs?**  
A: Yes, but the device will be CPU instead of MPS, resulting in significantly slower performance (5-10x slower).

**Q: Is my HuggingFace token secure?**  
A: Never commit tokens to Git! Use environment variables or the HuggingFace CLI login. Add config files with tokens to .gitignore.

**Q: How large is the model download?**  
A: Approximately 4GB for the base Qwen3-TTS model.

## 📚 Additional Resources

- **Official Qwen3-TTS Guide**: [Qwen3-TTS on Mac Mini M4](https://lingshunlab.com/ai/qwen3-tts-on-mac-mini-m4-the-ultimate-installation-optimization-guide)
- **HuggingFace Space**: [Qwen/Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS)
- **Alternative Implementation**: [esendjer/Q3-TTS](https://github.com/esendjer/Q3-TTS)
- **PyTorch MPS Documentation**: [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project follows the Qwen3-TTS model license. Please refer to the [official model page](https://huggingface.co/Qwen/Qwen3-TTS) for licensing details.

## 🙏 Acknowledgments

- Qwen Team for the excellent TTS model
- HuggingFace for hosting and infrastructure
- PyTorch team for MPS backend support

---

**Last Updated**: January 2026  
**Tested on**: macOS Sonoma/Sequoia with M4, PyTorch 2.5+