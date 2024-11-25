import numpy as np
import librosa
from tensorflow.keras.models import load_model
from joblib import load  
import os

model = load_model('best_model.keras')

script_dir = os.path.dirname(os.path.realpath(__file__))

scaler_path = os.path.join(script_dir, 'scaler.joblib')

print(f"Scaler path: {scaler_path}")

scaler = load(scaler_path)

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


def extract_features(file_path, n_mfcc=40, hop_length=512, n_fft=2048):
    audio, sample_rate = librosa.load(file_path, sr=None)
    if len(audio) < n_fft:
        n_fft = len(audio)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


def classify_audio(file_path):
    features = extract_features(file_path)
    features = np.array([features])
    
    features = scaler.transform(features)
    
    features = features[..., np.newaxis]

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)
    prediction_scores = prediction[0]
    
    return class_names[predicted_class[0]], prediction_scores
