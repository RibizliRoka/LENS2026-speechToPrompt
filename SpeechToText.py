import whisper

model = whisper.load_model("tiny")
result = model.transcribe("Testingaudio.m4a")
print(result["text"])