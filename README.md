#  Local Voice Intelligence System

An end-to-end, privacy-focused Voice-to-Voice AI Assistant. This system integrates Speech-to-Text, Large Language Model (LLM) reasoning, and Text-to-Speech into a seamless local pipeline.

## Key Features
- **Speech-to-Text (STT):** Powered by `Faster-Whisper` for low-latency, high-accuracy transcription.
- **Local Reasoning:** Uses a quantized `TinyLlama-1.1B` (GGUF) via `LangChain` for 100% on-device data processing.
- **Text-to-Speech (TTS):** Features Facebook's `MMS-TTS` for efficient and natural voice synthesis.
- **Intelligent Recording:** Implements auto-silence detection to stop recording after 3 seconds of inactivity.

## Technical Highlights
- **Architecture:** Modular pipeline designed to run on standard CPUs with < 2GB RAM.
- **Optimization:** Utilizes 4-bit quantization for the LLM and offline cache for the TTS model to ensure stability.
- **Audio Processing:** Real-time normalization using NumPy to maintain clear output levels.

## Setup
1. Install dependencies: `pip install -r requirements.txt`.
2. Place `tinyllama-q4_k_m.gguf` in the root directory.
3. Run `python main.py`.


