from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load your trained TensorFlow model
model = tf.keras.models.load_model('models/Leaf Deases(96,88).h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Preprocess the image
    # (You may need to resize, normalize, etc. based on your model requirements)

    # Perform prediction
    prediction = model.predict(image)

    # Return the result
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

