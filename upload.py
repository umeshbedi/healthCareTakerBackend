from flask import Flask, request, jsonify
import os

from imgProcess import imageProcess

app = Flask("__name__")

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle the image upload

def upload_image():


    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save the file to the specified upload folder
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the upload folder if it doesn't exist
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        try:
            return imageProcess(file.filename)
        except Exception as e:
            # Log the error to console and return error response
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Error during prediction'}), 500
         
        # jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200

    return jsonify({'error': 'File type not allowed'}), 400
