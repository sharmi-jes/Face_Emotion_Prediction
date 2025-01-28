from flask import Flask, request, render_template, send_from_directory
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model (replace with your actual model path)
model = load_model('E:\DEEP_LEARNING_PROJECTS\Emotion_Project\model (2).h5')

# Class labels for emotion prediction
# class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

train_dir=r'E:\DEEP_LEARNING_PROJECTS\Emotion_Project\train'
class_labels = sorted(os.listdir(train_dir))

# Predict function
def predict_img(img_path):
    img = load_img(img_path, target_size=(128, 128))  # Resize to 128x128
    img = img_to_array(img)
    
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image to the range [0, 1]

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]

    return class_labels[predicted_class_index], confidence

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # File handling
        file = request.files["file"]
        
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict emotion and confidence
            result, confidence = predict_img(file_location)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
