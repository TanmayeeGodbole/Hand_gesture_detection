from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json # <-- IMPORT Keras model loader
import mediapipe as mp
from function import mediapipe_detection, extract_keypoints # Make sure function.py is in the same folder
import os

app = Flask(__name__)

# --- Keras Model Setup (Replaced TFLite) ---
# We are now using the .json and .h5 files you said worked.
try:
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    print("Keras model loaded successfully.")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    print("Please make sure 'model.json' and 'model.h5' are in the correct directory.")
    model = None

# --- Constants ---
ACTIONS = np.array(['Hello', 'Left', 'Love','Ok', 'Right', 'Thankyou','Thumbdown', 'Thumbup','Peace','Fist','Loser','Up','Please'])
THRESHOLD = 0.8
SEQUENCE_LENGTH = 30

# MediaPipe hands setup
mp_hands = mp.solutions.hands

def gen_frames():
    """Video streaming generator function."""
    
    sequence = []
    sentence = []
    predictions = []
    accuracy_str = ""
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame.")
                break

            # --- Preprocessing: Your original crop logic ---
            cropframe = frame[40:400, 0:300]
            
            image, results = mediapipe_detection(cropframe, hands) 
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence[:] = sequence[-SEQUENCE_LENGTH:]

            # --- Prediction Logic (Using Keras model) ---
            if len(sequence) == SEQUENCE_LENGTH and model: # Check if Keras model is loaded
                try:
                    # Keras model.predict
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                    prediction = ACTIONS[np.argmax(res)]
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[-1] == np.argmax(res):
                        if res[np.argmax(res)] > THRESHOLD:
                            if not sentence or prediction != sentence[-1]:
                                sentence.append(prediction)
                                accuracy_str = f" {res[np.argmax(res)]*100:.0f}%"
                        
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                    
                    if res[np.argmax(res)] <= THRESHOLD and sentence:
                        sentence.pop()
                        accuracy_str = ""

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    sequence = []
            
            # --- Display Prediction on Frame ---
            cv2.rectangle(frame, (0, 40), (300, 400), (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence) + accuracy_str, (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- Stream the Frame ---
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                break

    cap.release()
    print("Video stream released.")


@app.route('/')
def index():
    """Home page."""
    print("Current working directory:", os.getcwd())
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the 'src' of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

