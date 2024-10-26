
from keras.models import load_model  # TensorFlow is required for Keras to work
from flask import jsonify, Flask
from PIL import Image, ImageOps 
import numpy as np
import os

app = Flask(__name__)

MODEL_DIR = os.path.join(app.root_path, 'model')
UPLOAD_DIR = os.path.join(app.root_path, 'uploads')

keraModel = os.path.join(MODEL_DIR, "keras_model.h5")
kerasText = os.path.join(MODEL_DIR, "labels.txt")



def imageProcess(filename):

    testImage = os.path.join(UPLOAD_DIR, filename)
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(keraModel, compile=False)

    # Load the labels
    class_names = open(kerasText, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(testImage).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    return jsonify({
        "class": class_name[2:],
        "confidenceScore":f"{confidence_score}"
    })

    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)



