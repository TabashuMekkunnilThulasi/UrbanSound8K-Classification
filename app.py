import streamlit as st
import numpy as np
import librosa
import os
from backend import classify_audio

os.makedirs('temp', exist_ok=True)

class_names = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


st.title('UrbanSound8K Audio Classification')
st.write('Upload a .wav file to classify the sound')


temp_file_path = None


uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file:
    
    temp_file_path = os.path.join('temp', uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_file_path, format='audio/wav')

    if st.button('Classify Audio'):
        result, scores = classify_audio(temp_file_path)
        st.write(f'Classification Result: {result}')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('**<h3>Top 5 Prediction Scores:</h3>**', unsafe_allow_html=True)
        scores *= 100 
        sorted_indices = np.argsort(scores)[::-1]   
        top_5_indices = sorted_indices[:5]  
        for i in top_5_indices:
            st.write(f'{class_names[i]}: {scores[i]:.2f}%')


if temp_file_path and os.path.exists(temp_file_path):
    os.remove(temp_file_path)
