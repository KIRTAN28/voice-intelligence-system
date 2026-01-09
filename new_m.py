from transformers import pipeline
import sounddevice as sd
import numpy as np

tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# 2. Generate Audio
text = "Hello, this is a real text to speech test. It is working now."
print(f"AI: {text}")
audio = tts(text)

# 3. FIX: Reshape and Normalize 
waveform = audio["audio"].flatten() 
sampling_rate = audio["sampling_rate"]

# Audio normalization 
if np.max(np.abs(waveform)) > 0:
    waveform = waveform / np.max(np.abs(waveform))


print("Playing Audio")
sd.play(waveform, sampling_rate)
sd.wait()
print("Done")


