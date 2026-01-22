import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import time

SAMPLE_RATE = 16000          # mic sample rate 
SILENCE_LIMIT = 3            # stop after 3 seconds of silence
SILENCE_THRESHOLD = 0.01     # silence sensitivity
MAX_TIME = 600               # we set max time for safty so that it automatically stop after 10 min

print("Start speaking")

# we are storing audio into audio data
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
        if volume < SILENCE_THRESHOLD:  # means if volume  of our voice is below the threshold then it ignore that voice and start counting if it reach to 3 second then it break the loop 
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_LIMIT:
                print("Silence detected")
                break
        else:
            silence_start = None
        # Safety stop
        if time.time() - start_time > MAX_TIME:
            print("Time limit reached")
            break


final_audio = np.concatenate(audio_data)
write("final_audio.wav", SAMPLE_RATE, final_audio)
print("Converting speech to text -> ")

model = WhisperModel(    # it converts speech into text 
    "base", 
    device="cpu",
    compute_type="int8"
)

segments, _ = model.transcribe("final_audio.wav")

print("Transcribed Text ")
tex =""
for segment in segments:
    tex+= segment.text +" "

print(tex)

#Part 2 here we generate answer of our text 
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

# Part 3  it convert text to into voice 
from transformers import pipeline
import sounddevice as sd
import numpy as np

tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")
text = input
print(f"AI: {text}")
audio = tts(text)

# Reshape and Normalize 
waveform = audio["audio"].flatten() 
sampling_rate = audio["sampling_rate"]

# Audio normalization 
if np.max(np.abs(waveform)) > 0:
    waveform = waveform / np.max(np.abs(waveform))

print("AI_response")
sd.play(waveform, sampling_rate)
sd.wait()
