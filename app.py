from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import base64
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model("cnn_pose_classifier.keras")
class_labels = [
    "adho mukh svanasana", "ashtanga namaskara", "ashwa sanchalanasana",
    "bhujangasana", "hasta utthanasana", "kumbhakasana",
    "padahastasana", "pranamasana"
]

movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet.signatures['serving_default']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(img_data)).convert('RGB')
    frame = np.array(image)

    # Resize and run MoveNet
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = tf.image.resize_with_pad(tf.convert_to_tensor(img_rgb), 192, 192)
    input_tensor = tf.expand_dims(tf.cast(img_input, dtype=tf.int32), axis=0)

    outputs = movenet(input_tensor)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    # Pose classification
    input_data = keypoints.flatten().reshape(1, 17, 3, 1)
    prediction = model.predict(input_data)
    pred_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        "pose": class_labels[pred_class],
        "confidence": confidence
    })

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Avoid GPU errors
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
