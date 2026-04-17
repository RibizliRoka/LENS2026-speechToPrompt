# I was thinking of using this to do the speech to text:
# https://realpython.com/python-speech-recognition/

# Once we get the text we can give it to gemini or something to make it into an actual
# text we can use with the model

# Also some potential cool factor ideas:

# -> Sticks - we can make a sort of sticks game with different colored blocks that are worth different
# amounts of points and we'd always ask it to pick up the next most valuable block and
# play against a person

# -> Recipie - robot picks up tiny food items which are ingredients to make something

# -> Organize by Color - robot picks up like colored tapes or blocks and puts them in
# different bins based on color, or this could be done with types of food, animals
# plants, etc 

import os
from google import genai
from PIL import Image
from google.genai import types
from dotenv import load_dotenv

def main():
    #speechText = speechToText()
    speechText = "Pick up the object that is shaped different"
    ourPrompt = speechTextToAI(speechText)
    print(ourPrompt)

def speechTextToAI(speechText):
    load_dotenv()
    client = genai.Client(api_key=os.getenv("apiKey"))
    img = Image.open("/home/laura-szabo/Downloads/robot.png")
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