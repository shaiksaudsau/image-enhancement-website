import os
from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
from collections import deque

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "static", "uploads")

# Store history of last 5 processed images in memory (per user session)
def add_to_history(image_url):
    if "history" not in session:
        session["history"] = []
    session["history"].insert(0, image_url)  # Add new image to front
    session["history"] = session["history"][:5]  # Keep only last 5

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    effect = request.form.get("effect")

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)

        # Apply selected enhancement
        if effect == "sharpen":
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            enhanced_img = cv2.filter2D(img, -1, kernel)

        elif effect == "grayscale":
            enhanced_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif effect == "contrast":
            alpha = 1.5
            beta = 30
            enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        elif effect == "denoise":
            enhanced_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        else:
            enhanced_img = img

        processed_filename = 'enhanced_' + file.filename
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

        cv2.imwrite(processed_path, enhanced_img)

        # URLs
        original_url = url_for('static', filename='uploads/' + file.filename)
        processed_url = url_for('static', filename='uploads/' + processed_filename)

        # Add to session history
        add_to_history(processed_url)

        return render_template(
            'result.html',
            original_image=original_url,
            processed_image=processed_url,
            effect=effect,
            history=session.get("history", [])
        )

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
