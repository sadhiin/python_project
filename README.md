# Python Project

# Flask Emotion Detection Web App

This is a simple web application built using Flask for real-time emotion detection from a webcam feed. The application uses a pre-trained model to predict emotions from the faces detected in the camera feed. Detected faces are outlined with a red square box, and the predicted emotion is displayed on top of the box.

## Features

- Open the device's camera and see real-time emotion predictions.
- Detected faces are outlined with a red square box, and the predicted emotion is displayed.
- Close the camera and stop predictions with the click of a button.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sadhiin/flask-emotion-detection.git
   cd flask-emotion-detection
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:

   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5000/predict` to access the Emotion Detection page.

## Dependencies

- Flask
- TensorFlow (for the model)
- OpenCV (for face detection)
- numpy
- base64 (for image handling)

## Files

- `app.py`: Flask backend that serves the web application, processes images, and makes predictions.
- `templates/predict.html`: HTML frontend for the Emotion Detection page.
- `haarcascade_frontalface_alt.xml`: Haar Cascade XML file for face detection.
- `EmoModel.h5`: Pre-trained emotion classification model.
- `utility.py`: Utility file for custom F1 score metric (if applicable).

## Acknowledgements

- The pre-trained model used in this project was trained on the AffecNet dataset.
- The Haar Cascade XML file for face detection is provided by OpenCV.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

