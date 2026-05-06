import numpy as np
import sounddevice as sd
from openwakeword.model import Model
import os
from pathlib import Path



SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
THRESHOLD = 0.3

MODEL_PATH = str(Path(__file__).parent / "Hey_Grip_ee.onnx")


      
def choose_working_input_device():
    devices = sd.query_devices()
    working_devices = []
    not_working_devices = []

    print("\nWorking input devices at 16000 Hz:\n")

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            try:
                sd.check_input_settings(
                    device=i,
                    channels=1,
                    samplerate=SAMPLE_RATE,
                    dtype="float32"
                )
                working_devices.append(i)
                print(f"{i}: {dev['name']} | default SR: {dev['default_samplerate']}")
            except Exception:
                not_working_devices.append((i,dev))

    print("\nNon-Working input devices at 16000 Hz:\n")
    for i,dev in not_working_devices:
        print(f"{i}: {dev['name']} | default SR: {dev['default_samplerate']}")
        

    while True:
        choice = int(input("\nType the number of the audio input device you want to use: "))
        if choice in working_devices:
            return choice
        print("That device does not work at 16000 Hz. Try another one.")


def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)

    audio_float = indata[:, 0]
    volume = np.sqrt(np.mean(audio_float ** 2))

    audio_int16 = np.clip(
        audio_float * 3.0 * 32767,
        -32768,
        32767
    ).astype(np.int16)

    prediction = model.predict(audio_int16)

    # Since you loaded only one custom model, just grab its score
    wakeword_name = list(prediction.keys())[0]
    score = list(prediction.values())[0]

    print(f"Volume: {volume:.4f} | {wakeword_name} score: {score:.3f}")

    if score > THRESHOLD:
        print(f"🔥 DETECTED: {wakeword_name}")

print("Model path:", MODEL_PATH)
print("Does",MODEL_PATH,"Exists?", os.path.exists(MODEL_PATH))


device_index = choose_working_input_device()
device_info = sd.query_devices(device_index)

print(f"\nUsing device #{device_index}: {device_info['name']}")

model = Model(
    wakeword_models=[MODEL_PATH],
    inference_framework="onnx"
)

with sd.InputStream(
    device=device_index,
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SIZE,
    callback=audio_callback
):
    print("\nListening for 'Hey Grip-ee'...")
    print("Press Ctrl+C to stop.\n")

    while True:
        pass