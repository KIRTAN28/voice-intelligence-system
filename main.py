import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import time

SAMPLE_RATE = 16000          # microphone sample rate
SILENCE_LIMIT = 3            # stop after 5 seconds of silence
SILENCE_THRESHOLD = 0.01     # silence sensitivity
MAX_TIME = 600                # safety limit (seconds)

print("Start speaking")

# Store all audio here
audio_data = []

silence_start = None
start_time = time.time()

# RECORD AUDIO 
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
) as stream :

    while True:
        data, _ = stream.read(int(0.1 * SAMPLE_RATE))

        audio_data.append(data)

        # Check volume
        volume = np.mean(np.abs(data))

        if volume < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_LIMIT:
                print(" Silence detected")
                break
        else:
            silence_start = None

        # Safety stop
        if time.time() - start_time > MAX_TIME:
            print(" Time limit reached")
            break


final_audio = np.concatenate(audio_data)
write("final_audio.wav", SAMPLE_RATE, final_audio)

# SPEECH TO TEXT
print("Converting speech to text...")

model = WhisperModel(    
    "base", 
    device="cpu",
    compute_type="int8"
)

segments, _ = model.transcribe("final_audio.wav")

print("\n Transcribed Text:")
tex =""
for segment in segments:
    tex+= segment.text +" "

print(tex)

#Part 2  {text --> Ai model (tinny llama) --->text }
from langchain_community.chat_models import ChatLlamaCpp
from langchain.messages import SystemMessage,HumanMessage

llm = ChatLlamaCpp(
    model_path=r"C:\open_source_model\tinyllama-q4_k_m.gguf",
    n_ctx=2048,
    verbose=False,
    n_threads=4
)
result = llm.invoke(tex)
input =result.content


# Part 3
from transformers import pipeline
import sounddevice as sd
import numpy as np

tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# 2. Generate Audio
text = input
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