import os
import cv2 as cv
import numpy as np
import whisper
from google import genai
from PIL import Image
from google.genai import types
from dotenv import load_dotenv

def main():
    #whisper stuff
    model = whisper.load_model("tiny")
    result = model.transcribe("Testingaudio.m4a")
    print(result["text"])

    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("No camera")
        exit()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("No frame")
            break
        
        cmd = input("q to quit, enter for picture: ").lower()
        if cmd != "q":
            cv.imwrite("output.png", frame)
            speechText = speechToText()
            speechText = "What is in the picture"
            ourPrompt = speechTextToAI(speechText)
            print(ourPrompt)
            # update txt file here
            print("got pic")
            os.remove("output.png")
        else:
            break
    cam.release()

def speechTextToAI(speechText):
    load_dotenv()
    client = genai.Client(api_key=os.getenv("apiKey"))
    img = Image.open("/home/laura-szabo/Code/School/Lens/speechToPrompt/output.png")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
        system_instruction="An image will be passed in along with a prompt asking the robot to pick a certain object or set of " \
        "objects up, however the robot's language model only understands very basic instructions like \'pick up the red block\'. What you need to" \
        "do is take the following prompt, and simplify it down for the robot using the picture as reference. Only return the modified prompt and nothing else."),
        contents=[img, speechText],
    )
    return response.text    


def speechToText(): #
    pass

if __name__ == "__main__":
    main()