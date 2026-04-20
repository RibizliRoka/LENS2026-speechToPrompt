import whisper
import threading
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard



startRecording_event = threading.Event()
stopRecording_event = threading.Event()
is_recording = False
audio_chunks = []

def on_press(key, injected):
    global is_recording
    try:
        #This is for the toggle on s, first if is to start recording and 
        #second if is end recording
        if key.char == 's' and not is_recording:
            is_recording = True
            startRecording_event.set()
        elif key.char == 's' and is_recording:
            is_recording = False
            stopRecording_event.set()

    except AttributeError:
        pass

def callback(indata, outdata, frames, time, status):
    global audio_chunks
    if status:
        print(status)
    audio_chunks.append(indata.copy())


print("Press s to start, and then press s to stop recording")

#Listener for Keyboard
listener = keyboard.Listener(on_press=on_press)
listener.start()

#Start recording until s is clicked
startRecording_event.wait()
print("Starting recording.....")

with sd.Stream(channels=2, callback=callback):
    #Stop recording when s is clicked again
    stopRecording_event.wait()

full_audio = np.concatenate(audio_chunks)

print("Stop recording.....")

#Stop the listener
listener.stop()

# Save as WAV file
fs = 44100  # Sample rate
write("output_sd.wav", fs, full_audio)
print("Audio saved to output_sd.wav")

# Translating the audio
print("Translating audio")
model = whisper.load_model("tiny")
result = model.transcribe("output_sd.wav")
print(result["text"])




