from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
import base64
from utility import f1_score

app = Flask(__name__)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the model
with tf.keras.utils.CustomObjectScope({"f1_score": f1_score}):
    model = load_model('../model/EmoModel.h5')
print(model.summary())


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # Extract the base64 image from the POST request
    image_data = request.form['imageBase64'].split(',')[1]
    decoded_img = base64.b64decode(image_data)
    nparr = np.fromstring(decoded_img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Use Haar cascades to detect faces
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = cv2.resize(img[y:y + h, x:x + w], (96, 96))
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, 96, 96, 3)

        # Make a prediction
        prediction = model.predict(face_img)
        emotion = np.argmax(prediction)
        classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        result = classes[emotion]
        print(result)
        # Draw a rectangle around detected face and display the emotion
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Convert the modified image with the box and text to base64
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)

    return jsonify(img_base64=jpg_as_text.decode("utf-8"))


if __name__ == '__main__':
    app.run(debug=True)
