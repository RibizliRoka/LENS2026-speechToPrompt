import whisper

model = whisper.load_model("tiny")
result = model.transcribe("output_sd.wav")
print(result["text"])


# import sounddevice as sd
# from scipy.io.wavfile import write

# # Recording parameters
# fs = 44100  # Sample rate
# seconds = 5  # Duration of recording

# print("Recording with SoundDevice...")
# # Record audio
# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
# sd.wait()  # Wait until recording is finished

# # Save as WAV file
# write("output_sd.wav", fs, myrecording)
# print("Audio saved to output_sd.wav")