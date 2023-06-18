from flask import Flask
from flask import render_template, request
import tensorflow as tf
import os
from PIL import Image
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
classes = ['hot dog','not hot dog'] # this is what we will see in html page

app = Flask(__name__)

def classify_image(image):
    
    image = image.resize((512, 512))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    classify_lite = interpreter.get_signature_runner('serving_default')
    predictions_lite = classify_lite(rescaling_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite).numpy()
    return "hot dog" if score_lite[0][0] > 0.5 else "not hot dog"



@app.route("/")
def home():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file)
        predictions = classify_image(image)
        # Save the image file temporarily
        print(file.filename)
        image_path = os.path.join('static/uploads', file.filename)
        image.save(image_path)
        
        return render_template('result.html', predictions=predictions, image_path=image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


