import streamlit as st
from groq import Groq
import base64
from PIL import Image as PILImage
import os
from dotenv import load_dotenv
import io

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq API client with API key
client = Groq(api_key=groq_api_key)

# New Model for Vision and Text
vision_model = 'llama-3.2-11b-vision-preview'
llama31_model = 'llama-3.1-70b-versatile'

# Function to resize image if too large
def resize_image(image, max_size=(800, 800)):
    """Resizes the image if it's larger than max_size."""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size)
    return image

# Function to encode image to base64 from memory
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Function to generate website description from wireframe using Llama 3.2 Vision
def image_to_test_case(client, model, base64_image, prompt):
    # Image and text prompt, without system prompt (as per the new model limitations)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ],
            }
        ],
        model=model
    )

    return chat_completion.choices[0].message.content

# Function to generate detailed test cases using Llama 3.1
def generate_detailed_test_cases(client, image_description):
    # No image is involved in this function, so the system prompt can be used here
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a QA test engineer. Based on the following wireframe or website screenshot, write detailed test cases in a structured format. Include sections such as Purpose, Preconditions, Steps, and Expected Results. Use the 'Given-When-Then' logic in the Steps section, but elaborate in full sentences.",
            },
            {
                "role": "user",
                "content": image_description,
            }
        ],
        model=llama31_model
    )
    
    return chat_completion.choices[0].message.content

# Streamlit app title
st.title("Test Case Generator - Powered by Groq")

# Image upload section
uploaded_image = st.file_uploader("Upload a website screenshot or wireframe", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Open the uploaded image using PIL
    image = PILImage.open(uploaded_image)
    
    # Resize the image if it's too large
    resized_image = resize_image(image)
    
    # Display the uploaded image
    st.image(resized_image, caption="Uploaded Wireframe or Website Screenshot (Resized)", use_column_width=True)
    
    # Encode the image to base64 (in-memory)
    base64_image = encode_image(resized_image)
    
    # Generate image description focusing on website elements (used internally, not displayed)
    description_prompt = "Describe this website or wireframe in detail, focusing on interactive elements like buttons, forms, and menus."
    image_description = image_to_test_case(client, vision_model, base64_image, description_prompt)
    
    # Generate detailed test cases based on the image description
    detailed_test_cases = generate_detailed_test_cases(client, image_description)
    
    # Display the generated test cases (without showing the website description)
    st.write("### Generated Test Cases")
    st.write(detailed_test_cases)
