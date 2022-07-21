"""
This is a Python API using the Flask framework
to interface Etno's AI with the application
"""

# Import the necessary modules
from flask import Flask, request, make_response
from flask_restx import Api, Resource
from PIL import Image
import numpy as np
import pickle
import io

# Set up variables
model = pickle.load(open('model.sav', 'rb'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Checks if the extension of the upload file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prepare the image for the model
def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((100, 100))
    img = np.array(img)
    img = img.reshape(1, -1)
    img = img / 255.0
    return img

app = Flask(__name__)
api = Api(app)

# The '/predict' route is used to predict the class of the image uploaded
@api.route('/predict', methods=['POST'])
class predict_image(Resource):
    def post(self):
        if 'image' not in request.files or not request.files.get('image'):
            return make_response("Please try again. The Image doesn't exist", 400)

        if not allowed_file(request.files['image'].filename):
            return make_response("Please try again. This file is not allowed", 415)

        file = request.files.get('image')
        img_bytes = file.read()
        img = prepare_image(img_bytes)

        return make_response(model.predict(img)[0], 200)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
