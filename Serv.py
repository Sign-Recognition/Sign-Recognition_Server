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

app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False
model = tf.keras.models.load_model('model_31.h5',custom_objects={'KerasLayer':hub.KerasLayer})
category_list=pd.Series(pd.read_csv('category_list_31.txt')['0'])

@app.route('/video', methods=['POST'])
def handle_request():
    files_ids = list(flask.request.files)
    for file_id in files_ids:
        print("\nRead Video ...")
        video_file = flask.request.files[file_id]
        data = np.frombuffer(video_file.read(), dtype=np.float64)
        data = data.reshape((1, 60, 224, 224, 3))
        # print(data)
        pred=model(data)
        pred=category_list[np.argmax(np.array(pred))]

    print("Send Complete!\n")
    return json.dumps(pred, ensure_ascii=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
