# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
import json
import os
from toBase64 import deBase64, toBase64
from detect import detect
import torch
import argparse
from utils.general import strip_optimizer


def goDetect():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='detect_img/input', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='detect_img/output', help='save results to project/name')
    parser.add_argument('--save-dir', type=str, default='detect_img/output', help='directory to save results')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)

"""
def getJsonFile(dataOrigin):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    json_path = dir_path + "\\detect_img\\output\\result.json"
    with open(json_path, 'r') as f:
        data = (json.load(f))

    return data
"""

# 启动flask程序
app = Flask(__name__)


@app.route('/get_json', methods=['POST', 'GET'])
def get_jsonData():
    # 获取request的json数据
    if request.json:
        for key, value in request.json.items():
            if key == "base64":
                print("Detect vary base64, start debase.")
                deBase64(value)
                break
        # 对解码后的图片进行检测
        print("Start Detect...")
        goDetect()

    # 获取输出的json
    # data = getJsonFile()

    # 将json数据传回网页
    return toBase64()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
