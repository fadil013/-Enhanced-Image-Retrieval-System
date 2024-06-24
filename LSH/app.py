from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np

# Import the LSH class and other necessary functions from your LSH code
from Q1 import LSH, euclidean_distance, cosine_similarity, load_cifar10_dataset, preprocess_and_extract_features

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # SQLite database
app.config['UPLOAD_FOLDER'] = r'C:\Users\abdul\OneDrive\Desktop\lab02\uploads'# Directory to store uploaded images
db = SQLAlchemy(app)

# Load CIFAR-10 training dataset and preprocess features
root_dir = r"C:\Users\HTS\Desktop\i221925-i221947-i221815_AbdullahNadeem-HarrisHassan-FadilAwan_A1 "

train_images, train_labels = load_cifar10_dataset(root_dir, train=True)
X_train = preprocess_and_extract_features(train_images)

# Instantiate LSH object and populate LSH tables with training data
num_tables = 5
num_functions = 5
num_buckets = 1000
num_dimensions = X_train.shape[1]  # Dimensionality of feature vectors
lsh = LSH(num_tables, num_functions, num_buckets, num_dimensions)
for vec in X_train:
    lsh.hash_vector(vec)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            # Save the uploaded file to the UPLOAD_FOLDER directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Load the uploaded image, preprocess it, and extract features
            uploaded_image = cv2.imread(file_path)
            uploaded_features = preprocess_and_extract_features([uploaded_image])
            # Execute query against LSH tables to find similar images
            neighbors = lsh.query(uploaded_features[0])
            # Render template to display similar images
            return render_template('similar_images.html', uploaded_image=file_path, similar_images=neighbors)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
