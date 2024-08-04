import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up directories
WORKING_DIR = '/path/to/working'  # Update with the path where the model and features are saved

# Load pre-trained VGG16 model
def load_vgg16_model():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model

# Load features from pickle
def load_features():
    with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    return features

# Load tokenizer
def load_tokenizer():
    with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load trained model
def load_captioning_model():
    model = load_model(os.path.join(WORKING_DIR, 'best_model(1).h5'))
    return model

# Load necessary resources
vgg_model = load_vgg16_model()
features = load_features()
tokenizer = load_tokenizer()
captioning_model = load_captioning_model()

# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption
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
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq ', '').replace(' endseq', '')

# Streamlit App
st.title('Image Captioning App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    # Extract features
    feature = vgg_model.predict(image, verbose=0)
    image_id = uploaded_file.name.split('.')[0]
    if image_id not in features:
        st.error('Image not found in features.')
    else:
        feature = features[image_id]
        max_length = 35  # Or dynamically determine if needed
        caption = predict_caption(captioning_model, feature, tokenizer, max_length)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write('Caption:', caption)
