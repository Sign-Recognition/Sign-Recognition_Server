import json
import os
import time
from collections import OrderedDict

import cv2
import flask
import werkzeug
import numpy as np
from flask import request
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from werkzeug import secure_filename
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False
model = tf.keras.models.load_model('model_31.h5',custom_objects={'KerasLayer':hub.KerasLayer})
category_list=pd.Series(pd.read_csv('category_list_31.txt')['0'])

@app.route('/video', methods=['POST'])
def handle_request():
    print("\nRead Video ...")
    video_file = flask.request.files["file"]
    video_file.save(secure_filename(video_file.filename))
    print("업로드 성공!")

    data = np.zeros((60, 224, 224, 3))
    cap = cv2.VideoCapture('temp.avi')

    for i in range(60):
        ret, data[i] = cap.read()
    cap.release()
    data = data.reshape((1, 60, 224,224,3))
    data /= 255.0
    pred = model(data)
    pred=category_list[np.argmax(np.array(pred))]
    print("Send Complete!\n")
    return json.dumps(pred, ensure_ascii=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)