import requests
import numpy as np
files = open("test.txt", "rb")
np_arr = np.random.randn(1, 60, 224, 224, 3)
print(np_arr)
upload= {'file':np_arr}
res = requests.post('http://127.0.0.1:5000/video', files=upload)
print(res.json())