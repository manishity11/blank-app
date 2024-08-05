import streamlit as st
import urllib.request
import os
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image

# Load models and tokenizer
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('best_model_inception.h5')

# Define a unique name for the Xception model to avoid conflicts
def get_xception_model():
    base_model = Xception(include_top=False, pooling="avg", weights="imagenet")
    base_model._name = "custom_xception"  # Assign a unique name to avoid conflicts
    return base_model

xception_model = get_xception_model()
max_length = 32

def extract_features_test(filename, model):
    try:
        image = Image.open(filename)
    except:
        st.error("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        return None
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def clearCaption(description):
    query = description
    stopwords = ['start', 'end']
    querywords = query.split()
    resultwords = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result

# Streamlit app
st.title("Image Captioning App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating caption...")
    
    # Save the uploaded image to a temporary file
    img_path = "temp.jpg"
    img.save(img_path)
    
    # Extract features and generate description
    photo = extract_features_test(img_path, xception_model)
    if photo is not None:
        description = generate_desc(model, tokenizer, photo, max_length)
        description = clearCaption(description)
        st.write("Caption:", description)
