## 🧠 Urban Sound Classification Using Deep Learning

### 🔍 Overview
This project aims to classify urban sound types (e.g., sirens, dog barks, drilling) using deep learning techniques. Built during my MSc in Data Science and Analytics, the model helps in identifying sound patterns for smart city and surveillance applications.

### 📊 Dataset
- **UrbanSound8K**: A dataset of 8,732 labeled sound excerpts (<=4s) of urban sounds from 10 classes.

### ⚙️ Technologies & Tools
- Python, TensorFlow, Keras, Librosa
- Jupyter Notebook

### 🧪 Model
- CNN (Convolutional Neural Network)
- Preprocessing with MFCC (Mel-Frequency Cepstral Coefficients)
- Achieved **high accuracy** and **low latency** suitable for real-time classification

### 📈 Results
- Model accuracy: ~90%
- Inference time: < 100ms

### 📂 Structure
```
UrbanSoundClassification/
│
├── audio_data/                  # Preprocessed sound files
├── notebooks/                   # Jupyter notebooks with model code
├── models/                      # Saved trained models
├── utils/                       # Helper functions
└── README.md                    # Project documentation
```

### 📌 Highlights
- Feature extraction using `librosa`
- Multiple deep learning models tested (CNN, LSTM, GRU)
- Selected CNN for best performance
- Integrated noise filtering for better prediction
