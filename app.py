import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# โหลด TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# รายชื่อประเภทขยะ
CLASS_NAMES = ['ขยะทั่วไป', 'ขยะรีไซเคิล', 'ขยะอินทรีย์']

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ฟังก์ชันพยากรณ์

def predict_waste_type(image_path):
    img = Image.open(image_path).resize((224, 224)).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_idx = np.argmax(output_data)
    return CLASS_NAMES[predicted_idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_waste_type(filepath)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
