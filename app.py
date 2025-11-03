import requests
import sys
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
model = load_model('face_emotionModel.h5')
model_url = https://drive.google.com/uc?export=download&id=1scmXLfFqUKUhF_ZbGR-zksVAA5MvXd1n

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img.reshape(1,48,48,1) / 255.0

    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]

    return render_template('index.html', prediction=emotion, image=file_path)

if __name__ == '__main__':
    app.run(debug=True)

