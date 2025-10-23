from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'bloodgroup_detection_secret_key'  # Change this for production!

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# List of blood groups (classes)
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Load the trained model once when the app starts
MODEL_PATH = 'saved_model/fingerprint_model.h5'
model = load_model(MODEL_PATH)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image before prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to CNN input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=[0, -1])  # Add batch and channel dimensions
    return img_array

# Home route
@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('upload'))

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'admin' and password == 'admin':  # Default credentials
            session['logged_in'] = True
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials!')

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Upload and predict route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected!')
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                processed_image = preprocess_image(filepath)
                prediction = model.predict(processed_image)

                predicted_class = BLOOD_GROUPS[np.argmax(prediction)]
                confidence = float(np.max(prediction) * 100)

                return render_template('result.html',
                                       filename=filename,
                                       blood_group=predicted_class,
                                       confidence=confidence)
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type! Allowed: png, jpg, jpeg')
            return redirect(request.url)

    return render_template('upload.html')

# Performance route
@app.route('/performance')
def performance():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('performance.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
