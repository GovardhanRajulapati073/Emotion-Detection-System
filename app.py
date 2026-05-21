from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("model/emotion_model.h5")

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json['image']

    # Remove base64 header
    encoded_data = data.split(',')[1]

    # Decode image
    np_data = np.frombuffer(base64.b64decode(encoded_data), np.uint8)

    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result = []

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))

        face = face / 255.0

        face = np.reshape(face, (1, 224, 224, 3))

        prediction = model.predict(face, verbose=0)

        emotion = emotion_labels[np.argmax(prediction)]

        confidence = float(np.max(prediction) * 100)

        result.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

    return jsonify(result)

@app.route('/health')
def health():
    return {
        "status": "running"
    }

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
