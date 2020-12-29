from flask import Flask
from flask_restful import Resource, Api, reqparse
import json
import numpy as np
import base64
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
# compression
import zlib
import codecs


app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('file', help = 'type error')

model = tf.keras.models.load_model('model_31.h5',custom_objects={'KerasLayer':hub.KerasLayer})
category_list=pd.Series(pd.read_csv('category_list_31.txt')['0'])

class Predict(Resource):
    def post(self):
        data = parser.parse_args()
        #print(data)
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }

        #img = open(data['imgb64'], 'r').read() # doesn't work
        img = data['file']
        data2 = img.encode()
        data2 = base64.b64decode(data2)
        data2 = zlib.decompress(data2)
        fdata = np.frombuffer(data2, dtype=np.float16)
        fdata = fdata.reshape((1, 60, 224, 224, 3))
        print(fdata)
        #fdata = fdata[:, :, :, :, [2, 1, 0]]
        # print(data)
        pred=model(fdata)
        pred=category_list[np.argmax(np.array(pred))]
        print("Send Complete!\n")
        return json.dumps(pred, ensure_ascii=False)


api.add_resource(Predict,'/video')

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000, threaded=True)
