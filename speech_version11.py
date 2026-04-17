import speech_recognition as sr
import pyaudio
import threading
from pynput import keyboard

# --- Configuration for PyAudio ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sample rate

# --- State Management ---
is_recording = threading.Event()
stop_program = threading.Event()

# --- Global State for Text Handling ---
current_transcription = None
transcription_lock = threading.Lock()

# --- NEW: This variable will hold the final, saved text for the main code ---
saved_text = None

# --- Continuous Recording and Transcription Function ---
def record_and_transcribe_continuous():
    """
    This function waits for a recording signal, records audio continuously,
    and then transcribes the entire segment when stopped.
    """
    global current_transcription
    recognizer = sr.Recognizer()

    while not stop_program.is_set():
        is_recording.wait()

        if stop_program.is_set():
            break

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("\n>>> Recording started...")
        frames = []

        while is_recording.is_set():
            try:
                data = stream.read(CHUNK)
                frames.append(data)
            except IOError:
                pass

        print(">>> Recording stopped. Processing audio...")
        stream.stop_stream()
        stream.close()
        p.terminate()

        if not frames:
            print("(Recording was empty.)")
            continue

        raw_audio_data = b''.join(frames)
        audio_data = sr.AudioData(raw_audio_data, RATE, p.get_sample_size(FORMAT))

        try:
            text = recognizer.recognize_google(audio_data)

            with transcription_lock:
                current_transcription = text

            print("\n--- Transcription Complete ---")
            print(text)
            print("\n>>> Press ENTER to save and continue, or 'l' for a new recording. <<<")

        except sr.UnknownValueError:
            print("Google could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google; {e}")


# --- Keyboard Listener Function ---
def on_press(key):
    """
    Toggles recording on 'l'.
    On 'Enter': saves text and STOPS the listener, returning control to main.
    On 'Esc': exits without saving.
    """
    global current_transcription
    global saved_text  # Declare that we will modify this global variable

    # Handle 'l' for recording
    try:
        if key.char == 'l':
            if is_recording.is_set():
                is_recording.clear()
            else:
                is_recording.set()
    except AttributeError:
        pass

    # Handle Enter key press
    if key == keyboard.Key.enter:
        with transcription_lock:
            if current_transcription:
                # --- THE KEY CHANGE ---
                # 1. Save the text to the global variable
                saved_text = current_transcription
                current_transcription = None

                print("\nText saved!")

                # 2. Signal the speech thread to stop
                stop_program.set()
                is_recording.set()  # Unblock if waiting

                # 3. Stop the keyboard listener by returning False
                return False
            else:
                pass  # No text to save, do nothing

    # Handle 'Esc' to exit without saving
    if key == keyboard.Key.esc:
        print("Exit command received. Shutting down without saving.")
        stop_program.set()
        is_recording.set()
        return False


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Continuous Speech-to-Text Program ---")
    print("Press 'l' to start recording.")
    print("Press 'l' again to stop and transcribe.")
    print("Press 'ENTER' to save the text and continue to the main code.")
    print("Press 'Esc' to exit without saving.")

    # Start the speech recognition thread
    speech_thread = threading.Thread(target=record_and_transcribe_continuous, daemon=True)
    speech_thread.start()

    # Start the keyboard listener. This will BLOCK here until Enter or Esc is pressed.
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Wait for the speech thread to clean up
    speech_thread.join(timeout=2)


    if saved_text:
        print(saved_text)

    else:
        print("\nNo text was saved (you may have pressed Esc).")

    print("\nProgram finished.")
    