import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response, make_response
import os
import threading
import time

app = Flask(__name__)

# Global variables - will be initialized on first request
model = None
face_cascade = None
cap = None
camera_lock = threading.Lock()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def init_model():
    """Initialize model and cascade on first request"""
    global model, face_cascade
    if model is None:
        print("Loading model...")
        model = load_model("model/emotion_model.h5")
        print("Model loaded!")
    if face_cascade is None:
        print("Loading face cascade...")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("Face cascade loaded!")

def init_camera_with_timeout():
    """Initialize camera with timeout to prevent hanging"""
    global cap
    
    if cap is not None and cap.isOpened():
        return cap
    
    with camera_lock:
        print("Attempting to initialize camera...")
        try:
            # Try different camera indices in case 0 doesn't work
            for camera_index in [0, 1, -1]:
                print(f"Trying camera index: {camera_index}")
                cap = cv2.VideoCapture(camera_index)
                
                # Set a timeout using a thread
                def check_camera(video_cap, timeout=5):
                    start = time.time()
                    while time.time() - start < timeout:
                        ret, frame = video_cap.read()
                        if ret:
                            return True
                        time.sleep(0.1)
                    return False
                
                if check_camera(cap, timeout=3):
                    print(f"Camera initialized successfully on index {camera_index}!")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
                else:
                    cap.release()
                    cap = None
                    print(f"Camera index {camera_index} failed or timed out")
            
            print("ERROR: Could not access any camera!")
            return None
        except Exception as e:
            print(f"ERROR: Exception while initializing camera: {e}")
            import traceback
            traceback.print_exc()
            return None

def annotate_emotion(frame):
    """Annotate the frame with detected emotions and bounding boxes."""
    global model, face_cascade
    if model is None or face_cascade is None:
        init_model()

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (224, 224))
                face = face / 255.0
                face = np.reshape(face, (1, 224, 224, 3))

                prediction = model.predict(face, verbose=0)
                emotion = emotion_labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_text = f"{emotion} ({confidence:.1f}%)"
                cv2.putText(frame, label_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
    except Exception as e:
        print(f"Error in face detection: {e}")
    return frame

def detect_emotion_frame():
    """Generate frames with emotion detection"""
    global cap, model, face_cascade
    
    init_model()
    cap = init_camera_with_timeout()
    
    if cap is None:
        print("ERROR: Camera not available - yielding error frame")
        # Generate an error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ERROR: Camera not available", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_frame, "Check if webcam is connected", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    frame_count = 0
    error_count = 0
    max_consecutive_errors = 10
    
    while error_count < max_consecutive_errors:
        try:
            with camera_lock:
                if cap is None or not cap.isOpened():
                    print("Camera was disconnected")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    error_count += 1
                    print(f"Failed to read frame ({error_count}/{max_consecutive_errors})")
                    if error_count >= max_consecutive_errors:
                        break
                    continue
            
            # Reset error count on successful read
            error_count = 0
            frame_count += 1
            
            # Make a copy to avoid issues with frame modification
            frame = frame.copy()
            frame = annotate_emotion(frame)

            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame to JPEG with quality
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                print("ERROR: Failed to encode frame")
                error_count += 1
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Send frame with proper boundary
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                       frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error sending frame: {e}")
                break
                
        except Exception as e:
            print(f"Error in frame generation: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            if error_count >= max_consecutive_errors:
                break
    
    print("Video stream ended")
    cap.release()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detect_emotion_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    """Return a single annotated frame for browsers that need a fallback."""
    init_model()
    cap = init_camera_with_timeout()

    if cap is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ERROR: Camera unavailable", (40, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 50, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response

    with camera_lock:
        ret, frame = cap.read()

    if not ret or frame is None:
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ERROR: Frame capture failed", (40, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 50, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response

    frame = annotate_emotion(frame)
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'ok'}, 200

if __name__ == '__main__':
    print("Starting Emotion Detection Web App...")
    print("Visit http://localhost:5000 in your browser")
    print("Make sure your webcam is connected!")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
