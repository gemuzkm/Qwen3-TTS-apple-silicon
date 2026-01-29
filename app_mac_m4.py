# coding=utf-8

# Qwen3-TTS Gradio Demo for Mac M4 (Optimized)
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)
# Оптимизировано для локального запуска на Mac с M4 чипом

import os
import sys
import logging
import gradio as gr
import numpy as np
import torch
import soundfile as sf
from huggingface_hub import snapshot_download, login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= MAC M4 HARDWARE DETECTION =============
def get_device():
    """Auto-detect hardware and return appropriate device."""
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("✅ Using Apple Metal Performance Shaders (MPS) - Mac M4 detected")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("✅ Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        logger.warning("⚠️  Using CPU - this will be slow!")
    return device

def sync_device():
    """Hardware-aware synchronization for proper device handling."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()  # Critical for M4 stability!
    except Exception as e:
        logger.debug(f"Sync warning: {e}")

# ============= AUTHENTICATION =============
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("✅ HuggingFace authentication successful")
    except Exception as e:
        logger.warning(f"⚠️  HuggingFace login failed: {e}")
else:
    logger.warning("⚠️  HF_TOKEN not set - using read-only access")

# ============= GLOBAL MODEL CACHE =============
loaded_models = {}
DEVICE = get_device()

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]

def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"
    logger.info(f"📥 Downloading model: {model_name}")
    return snapshot_download(model_name)

def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models
    key = (model_type, model_size)
    
    if key not in loaded_models:
        logger.info(f"🔄 Loading model: {model_type} ({model_size})")
        try:
            from qwen_tts import Qwen3TTSModel
            
            model_path = get_model_path(model_type, model_size)
            
            # MAC M4 OPTIMIZED LOADING
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,      # M4 supports bfloat16 natively
                attn_implementation="sdpa",      # Use SDPA (Flash Attention not compatible with M4)
                device_map=DEVICE,               # Auto-use MPS on Mac M4
                token=HF_TOKEN,
            )
            logger.info(f"✅ Model loaded successfully on {DEVICE}")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    return loaded_models[key]

# ============= AUDIO UTILITIES =============
def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)
    
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    
    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    
    return y

def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None
    
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    
    return None

# ============= SPEAKER AND LANGUAGE CHOICES =============
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", 
    "Serena", "Sohee", "Uncle_fu", "Vivian"
]

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", 
    "French", "German", "Spanish", "Portuguese", "Russian"
]

# ============= GENERATION FUNCTIONS (MAC OPTIMIZED) =============
def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "❌ Error: Text is required."
    
    if not voice_description or not voice_description.strip():
        return None, "❌ Error: Voice description is required."
    
    try:
        logger.info("🎨 Starting Voice Design generation...")
        tts = get_model("VoiceDesign", "1.7B")
        
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        sync_device()  # Ensure MPS synchronization
        logger.info("✅ Voice Design generation completed")
        return (sr, wavs[0]), "✅ Voice design generation completed successfully!"
    
    except Exception as e:
        logger.error(f"❌ Voice Design error: {type(e).__name__}: {e}")
        return None, f"❌ Error: {type(e).__name__}: {str(e)[:200]}"

def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "❌ Error: Target text is required."
    
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "❌ Error: Reference audio is required."
    
    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "❌ Error: Reference text is required when 'Use x-vector only' is not enabled."
    
    try:
        logger.info(f"🎤 Starting Voice Clone generation ({model_size})...")
        tts = get_model("Base", model_size)
        
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        
        sync_device()  # Ensure MPS synchronization
        logger.info("✅ Voice Clone generation completed")
        return (sr, wavs[0]), "✅ Voice clone generation completed successfully!"
    
    except Exception as e:
        logger.error(f"❌ Voice Clone error: {type(e).__name__}: {e}")
        return None, f"❌ Error: {type(e).__name__}: {str(e)[:200]}"

def generate_custom_voice(text, language, speaker, instruct, model_size):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "❌ Error: Text is required."
    
    if not speaker:
        return None, "❌ Error: Speaker is required."
    
    try:
        logger.info(f"🎙️ Starting CustomVoice generation ({speaker}, {model_size})...")
        tts = get_model("CustomVoice", model_size)
        
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        sync_device()  # Ensure MPS synchronization
        logger.info("✅ CustomVoice generation completed")
        return (sr, wavs[0]), "✅ Generation completed successfully!"
    
    except Exception as e:
        logger.error(f"❌ CustomVoice error: {type(e).__name__}: {e}")
        return None, f"❌ Error: {type(e).__name__}: {str(e)[:200]}"

# ============= GRADIO UI =============
def build_ui():
    """Build the Gradio interface."""
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )
    
    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    .info-box {background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 10px 0;}
    """
    
    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo (Mac M4)") as demo:
        gr.Markdown(f"""
        # 🎙️ Qwen3-TTS Demo (Mac M4 Optimized)
        
        **Device:** {DEVICE.upper()} | **PyTorch:** {torch.__version__}
        
        A unified Text-to-Speech demo featuring three powerful modes:
        
        - **🎨 Voice Design**: Create custom voices using natural language descriptions
        - **🎤 Voice Clone (Base)**: Clone any voice from a reference audio
        - **🎙️ TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions
        
        Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
        
        **⚠️ Note:** First generation may take longer as models are loaded. Subsequent calls will be faster.
        """)
        
        with gr.Tabs():
            # Tab 1: Voice Design
            with gr.Tab("🎨 Voice Design"):
                gr.Markdown("### Create Custom Voice with Natural Language Description")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible!",
                        )
                        
                        design_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                        
                        design_instruct = gr.Textbox(
                            label="Voice Description",
                            lines=3,
                            placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic.",
                        )
                        
                        design_btn = gr.Button("🎨 Generate with Custom Voice", variant="primary")
                    
                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)
                
                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status],
                )
            
            # Tab 2: Voice Clone
            with gr.Tab("🎤 Voice Clone (Base)"):
                gr.Markdown("### Clone Voice from Reference Audio")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio (Upload a voice sample to clone)",
                            type="numpy",
                        )
                        
                        clone_ref_text = gr.Textbox(
                            label="Reference Text (Transcript of the reference audio)",
                            lines=2,
                            placeholder="Enter the exact text spoken in the reference audio...",
                        )
                        
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (faster, lower quality)",
                            value=False,
                        )
                    
                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="Target Text (Text to synthesize with cloned voice)",
                            lines=4,
                            placeholder="Enter the text you want the cloned voice to speak...",
                        )
                        
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            
                            clone_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        
                        clone_btn = gr.Button("🎤 Clone & Generate", variant="primary")
                
                with gr.Row():
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)
                
                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size],
                    outputs=[clone_audio_out, clone_status],
                )
            
            # Tab 3: CustomVoice TTS
            with gr.Tab("🎙️ TTS (CustomVoice)"):
                gr.Markdown("### Text-to-Speech with Predefined Speakers")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities.",
                        )
                        
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                            
                            tts_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                        
                        tts_instruct = gr.Textbox(
                            label="Style Instruction (Optional)",
                            lines=2,
                            placeholder="e.g., Speak in a cheerful and energetic tone",
                        )
                        
                        tts_model_size = gr.Dropdown(
                            label="Model Size",
                            choices=MODEL_SIZES,
                            value="1.7B",
                            interactive=True,
                        )
                        
                        tts_btn = gr.Button("🎙️ Generate Speech", variant="primary")
                    
                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)
                
                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size],
                    outputs=[tts_audio_out, tts_status],
                )
        
        gr.Markdown("""
        ---
        
        ### 🔧 Mac M4 Optimizations
        
        This version includes:
        - ✅ Auto-detection of Apple Metal Performance Shaders (MPS)
        - ✅ Hardware-aware device synchronization
        - ✅ bfloat16 support optimized for M4
        - ✅ SDPA attention (Flash Attention 2 is incompatible with M4)
        - ✅ Proper error handling and logging
        
        **Installation for Mac M4:**
        ```bash
        conda create -n qwen3-tts python=3.12 -y
        conda activate qwen3-tts
        pip install -U qwen-tts gradio torch
        python app_mac_m4.py
        ```
        """)
        
        return demo

if __name__ == "__main__":
    logger.info("🚀 Starting Qwen3-TTS Demo (Mac M4 Optimized)")
    logger.info(f"📊 System Info: Device={DEVICE}, PyTorch={torch.__version__}")
    
    demo = build_ui()
    
    # Launch with proper settings for Mac
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
    )
