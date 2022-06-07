import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
import imp
import pathlib
import io
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import sys

class Resquest(BaseHTTPRequestHandler):
    def handler(self):
        print("data:", self.rfile.readline().decode())
        self.wfile.write(self.rfile.readline())

    def do_GET(self):
        print(self.requestline)
        # if self.path != '/hello':
        #     self.send_error(404, "Page not Found!")
        #     return
        image_np = load_image_into_numpy_array('./a.webp')
        detector_output = detector(image_np)
        class_ids = detector_output["detection_classes"]
        # 输出图片中包含的类别 例如人 房子
        print(class_ids)
        data = {
            'result_code': '1',
            'result_desc': 'Success',
            'timestamp': str(time.time()),
            'data': {'predict_result': str(class_ids) }
        }
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        print(self.headers)
        print(self.command)
        req_datas = self.rfile.read(int(self.headers['content-length'])) #重点在此步!
        print("req_datas is : %s" % req_datas.decode())

        # image_np = load_image_into_numpy_array('./a.webp')
        image_np = load_image_into_numpy_array(req_datas.decode())
        detector_output = detector(image_np)
        class_ids = detector_output["detection_classes"]
        # 输出图片中包含的类别 例如人 房子
        print(class_ids)
        data = {
            'result_code': '1',
            'result_desc': 'Success',
            'timestamp': str(time.time()),
            'data': {'predict_result': str(class_ids) }
        }
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

def _init_module():
    file_handle, pathname, desc = None, None, None
    file_handle, pathname, desc = imp.find_module("tensorflow", pathname)
    imp.load_module("tensorflow", file_handle, pathname, desc)
    file_handle, pathname, desc = None, None, None
    file_handle, pathname, desc = imp.find_module("numpy", pathname)
    imp.load_module("numpy", file_handle, pathname, desc)

def load_image_into_numpy_array(path):
    image = None
    if (path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    print(image.size)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)

detector = tf.compat.v1.saved_model.load_v2('./faster_rcnn_inception_resnet_v2_640x640_1')
detector._is_hub_module_v1 = False

if __name__ == '__main__':
#    _init_module()
    host = ('', 9001)
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()

