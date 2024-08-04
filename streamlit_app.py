import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('img_caption_model.h5')

# Load the ResNet50 model with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet')
model_fe = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

max_length = 34  # This should be set to the maximum length used in your training

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def extract_features(image):
    image = preprocess_image(image)
    feature = model_fe.predict(image, verbose=0)
    return feature

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

st.title("Image Caption Generator")
st.write("Upload an image to generate a caption")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating caption...")
    
    feature = extract_features(image)
    caption = predict_caption(model, feature, tokenizer, max_length)
    
    st.write("Caption:")
    st.write(caption)
