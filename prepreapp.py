import numpy as np
import cv2
from utility import f1_score
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the pre-trained emotion prediction model
with tf.keras.utils.CustomObjectScope({"f1_score": f1_score}):
    model = tf.keras.models.load_model('EmoModel.h5')

# List of emotion labels
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = None


def generate_frames():
    global camera
    while True:
        if camera is not None:
            ret, frame = camera.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)
                emotion_prob = model.predict(face)
                emotion = emotion_labels[np.argmax(emotion_prob)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        else:
            break


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/open_camera')
def open_camera():
    global camera
    camera = cv2.VideoCapture(0)
    return "Camera opened"


@app.route('/close_camera')
def close_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Camera closed"


@app.route('/predict')
def predict():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
