import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import Audio

st.title("Image to Audio Text Generation")

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKENS = os.getenv("api_token")

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]['generated_text']
    st.text("Generated Text from Image:")
    st.write(text)

    st.image(url, use_column_width=True)
    return text

# Get the image URL from the user
image_url = st.text_input("Enter the URL of the image:")
if st.button("Generate Text from Image"):
    image_caption = img2text(image_url)

    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": "Bearer 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"}

    def query(prompt, max_new_tokens=200):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    data = query(image_caption, max_new_tokens=250)
    generated_text = data[0].get("generated_text", "")
    st.text("Generated Text:")
    st.write(generated_text)

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKENS}"}

    def generate_and_play_audio(text, sampling_rate=22050):
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)
        audio = response.content

        # Create an Audio object to play the binary audio data
        return Audio(audio, rate=sampling_rate)

    # if st.button("Generate and Play Audio"):
    text_to_speak = generated_text
    audio_object = generate_and_play_audio(text_to_speak)
    st.write("Audio Response:")
    st.audio(audio_object.data, format="audio/wav")

