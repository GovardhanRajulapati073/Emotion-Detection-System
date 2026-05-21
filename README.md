
Emotion Detection Project

This project can now run as a browser-based web application using Flask. The live emotion detection stream is rendered in your browser instead of an OpenCV desktop window.

## Run the web app

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. Open your browser and visit:
   ```bash
   http://localhost:5000
   ```

## Notes

- The app uses your local webcam and serves the live video stream in the browser.
- If the webcam is unavailable, the UI will show a helpful message and allow you to retry the stream.
- `live_emotion.py` remains available for the original OpenCV window-based version.

## Existing files

- `app.py` — Flask web server that streams annotated emotion detection.
- `train_model.py` — Train the model using the dataset.
- `model/emotion_model.h5` — Pretrained emotion classification model.
- `dataset/train/...` — Training image folders.
