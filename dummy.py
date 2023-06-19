import tensorflow as tf
import cv2
from endpoint import predict_custom_trained_model_sample
image = cv2.imread('images\download.jpg')
image = cv2.resize(image, (512,512))

TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model

img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
"""interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
print(interpreter.get_signature_list())
classify_lite = interpreter.get_signature_runner('serving_default')
predictions_lite = classify_lite(rescaling_input=img_array)['dense_1']
score_lite = tf.nn.softmax(predictions_lite)
print(score_lite[0][0])"""
predict_custom_trained_model_sample(
    project="1016522667623",
    endpoint_id="39226176832536576",
    location="us-central1",
    instances={ "instance_key_1": img_array}
)