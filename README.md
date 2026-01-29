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

3. **Login with your token**

   The token is stored in your user configuration and persists across sessions. You only need to login once.

   ```bash
   # Install HuggingFace CLI (if not already installed)
   pip install huggingface_hub
   
   # Login with your token
   huggingface-cli login
   # Paste your token when prompted (e.g., hf_YourTokenHere)
   ```

   This saves the token to `~/.cache/huggingface/token` and you won't need to provide it again.

   **Alternative: Login in Python script**
   ```python
   from huggingface_hub import login
   login(token="hf_YourTokenHere")  # Only needed once
   ```

4. **Accept model license**
   - Visit [Qwen3-TTS model page](https://huggingface.co/Qwen/Qwen3-TTS)
   - Click "Agree and access repository" if prompted

## 🚀 Quick Start

### Installation Method 1: Using venv (Recommended)

```bash
# 1. Install system dependencies first
brew install portaudio ffmpeg sox

# 2. Clone this repository
git clone https://github.com/gemuzkm/Qwen3-TTS-apple-silicon.git
cd Qwen3-TTS-apple-silicon

# 3. Create virtual environment in the repository directory
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

# 6. Login to HuggingFace (one-time setup)
huggingface-cli login
# Paste your token when prompted

# 7. Run the application
python app_mac_m4.py
```

### Installation Method 2: Using Conda

```bash
# 1. Install system dependencies first
brew install portaudio ffmpeg sox

# 2. Clone this repository
git clone https://github.com/gemuzkm/Qwen3-TTS-apple-silicon.git
cd Qwen3-TTS-apple-silicon

# 3. Create conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# 4. Install all packages via pip (even with conda)
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install qwen-tts gradio transformers==4.57.3 accelerate==1.12.0

# 5. Login to HuggingFace (one-time setup)
huggingface-cli login
# Paste your token when prompted

# 6. Run the application
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
)
```

Run with float32:
```bash
export QWEN_DTYPE="float32"
python app_mac_m4.py
```

## 📁 Project Structure

After cloning and setup, your directory will look like:

```
Qwen3-TTS-apple-silicon/
├── app_mac_m4.py           # Main application file
├── README.md               # This file
├── .gitignore             # Git ignore rules
└── venv/                  # Virtual environment (not tracked in git)
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
- "Access to model denied" error

**Solution:**
```bash
# Check if you're logged in:
huggingface-cli whoami

# If not logged in or token expired, login again:
huggingface-cli login
# Paste your token when prompted

# Verify token is saved:
ls -la ~/.cache/huggingface/token
```

**Note**: You don't need to use `export HF_TOKEN` because `huggingface-cli login` stores the token in your user configuration permanently.

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
# Use HuggingFace mirror (if available in your region)
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

#### 6. "Repository not found or access denied"

**Solution:**
1. Make sure you accepted the model license at [Qwen3-TTS page](https://huggingface.co/Qwen/Qwen3-TTS)
2. Verify you're logged in: `huggingface-cli whoami`
3. Re-login if needed: `huggingface-cli login`

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

### Verify HuggingFace authentication:

```bash
# Check login status
huggingface-cli whoami

# Should display your username if logged in
# If not, run: huggingface-cli login
```

### Model loading test:

```python
from qwen_tts import Qwen3TTSModel
import torch

# Token not needed here - already logged in via CLI
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS",
    torch_dtype=torch.bfloat16,
    device_map="mps"
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

**Q: Do I need to set HF_TOKEN environment variable?**  
A: No! Using `huggingface-cli login` stores the token in your user configuration (`~/.cache/huggingface/token`) permanently. You don't need to export or set any environment variables.

**Q: Why use bfloat16 instead of float32?**  
A: Apple Silicon has hardware-accelerated bfloat16 support, resulting in 10-15% faster inference and reduced memory usage while maintaining excellent audio quality.

**Q: Why SDPA instead of Flash Attention 2?**  
A: Flash Attention is not compatible with Metal Performance Shaders (MPS). SDPA (Scaled Dot-Product Attention) is PyTorch's standard attention mechanism that works across all platforms.

**Q: Do I need portaudio and sox?**  
A: Yes, these are system-level audio processing libraries required by Python audio packages (soundfile, librosa). Without them, you'll encounter import errors.

**Q: Can this run on Intel Macs?**  
A: Yes, but the device will be CPU instead of MPS, resulting in significantly slower performance (5-10x slower).

**Q: Where is my HuggingFace token stored?**  
A: After running `huggingface-cli login`, the token is securely stored in `~/.cache/huggingface/token` and used automatically by all HuggingFace libraries.

**Q: How large is the model download?**  
A: Approximately 4GB for the base Qwen3-TTS model.

**Q: Can I use this repository for other projects?**  
A: Yes! After cloning, the `venv/` directory is excluded from git (see .gitignore), so each clone can have its own independent environment.

## 📚 Additional Resources

- **Official Qwen3-TTS Guide**: [Qwen3-TTS on Mac Mini M4](https://lingshunlab.com/ai/qwen3-tts-on-mac-mini-m4-the-ultimate-installation-optimization-guide)
- **HuggingFace Space**: [Qwen/Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS)
- **Alternative Implementation**: [esendjer/Q3-TTS](https://github.com/esendjer/Q3-TTS)
- **PyTorch MPS Documentation**: [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- **HuggingFace CLI Documentation**: [HuggingFace Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

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