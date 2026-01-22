from transformers import pipeline
import sounddevice as sd
import numpy as np

tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")
text = "Hello this is working now"
print("AI: {text}")
audio = tts(text)

# Reshape and Normalize 
waveform = audio["audio"].flatten() 
sampling_rate = audio["sampling_rate"]

# Audio normalization 
if np.max(np.abs(waveform)) > 0:
    waveform = waveform / np.max(np.abs(waveform))


print("Playing Audio")
sd.play(waveform, sampling_rate)
sd.wait()
print("Done")


