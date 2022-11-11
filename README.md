# 一、赛题
赛题A4-林业有害生物智能识别
![image-20221111215921742](https://adrainty-1301677055.cos.ap-shanghai.myqcloud.com/image/20221111215956image-2022111121592174220.png)

# 二、训练集的训练

## 1、创建虚拟环境

```bash
conda create -n torch107 python=3.7
```

安装完成后，输入

```bash
activate torch107
```

## 2、安装CUDA

[安装连接点这里](https://developer.nvidia.com/cuda-10.1-download-archive-update2)

## 3、安装PyTorch

[安装连接点这里](http://pytorch.org)

## 4、查看CUDA是否可用

```python
import torch
torch.cuda.is_available()
```

出现True结果即可

## 5、下载源码和依赖库

https://github.com/ultralytics/yolov5

下载后解压，在目录内打开cmd并激活环境：

~~~bash
activate torch107
~~~

安装依赖库

~~~bash
pip install -r requirements.txt
~~~

## 6、数据标注

数据标注需要用到labelimg，用pip安装它

```bash
pip install labelimg
```

安装好后用cmd输入labelimg即可打开标注软件

- 在view中选择Auto Save Mode

- 在Change Save Dir中选择要存放的目录

- Open Dir打开需要标注图片的位置

标注完成后，每张图像会生成对应的xml标注文件

我们将图像和数据统一放置到源码目录的VOCData文件夹下。
 其中jpg文件放置在VOCData/images下，xml放置在VOCData/Annotations下：

## 7、数据预处理

创建 split.py 文件，内容如下：

~~~python
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', default='VOCData/Annotations', type=str, help='input xml label path')
parser.add_argument('--txt_path', default='VOCData/labels', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
~~~

运行结束后，可以看到VOCData/labels下生成了几个txt文件

然后新建 txt2yolo_label.py 文件用于将数据集转换到yolo数据集格式

~~~python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ['face', 'normal', 'phone', 'write',
           'smoke', 'eat', 'computer', 'sleep']


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    # try:
        in_file = open('VOCData/Annotations/%s.xml' % (image_id), encoding='utf-8')
        out_file = open('VOCData/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')
    # except Exception as e:
    #     print(e, image_id)


wd = getcwd()
for image_set in sets:
    if not os.path.exists('VOCData/labels/'):
        os.makedirs('VOCData/labels/')
    image_ids = open('VOCData/labels/%s.txt' %
                     (image_set)).read().strip().split()
    list_file = open('VOCData/%s.txt' % (image_set), 'w')
    for image_id in tqdm(image_ids):
        list_file.write('VOCData/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
~~~

把classes改成自己标注的标签

转换后可以看到VOCData/labels下生成了每个图的txt文件

在data文件夹下创建myvoc.yaml文件，内容如下：

~~~yaml
train: VOCData/train.txt
val: VOCData/val.txt

# number of classes
nc: 8

# class names
names: ["face", "normal", "phone", "write", "smoke", "eat", "computer", "sleep"]
~~~

其中，nc改成标签数，names改成自己的标签

## 8、下载预训练模型

https://github.com/ultralytics/yolov5

将下载好的预训练模型放到weights文件夹下

## 9、开始训练

修改models/yolov5m.yaml下的类别数，即nc的数目

然后在cmd中输入：

~~~bash
python train.py --img 640 --batch 4 --epoch 300 --data ./data/myvoc.yaml --cfg ./models/yolov5m.yaml --weights weights/yolov5m.pt --workers 0
~~~

即可开始训练

训练好的模型会放在runs\train中

## 10、数据检测

模型推理测试在cmd中输入

```bash
python detect.py --source data/images --weights last.pt --conf 0.25
```

如果需要更改可以参照

```python
	"""
    weights:训练的权重
    source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    output:网络预测之后的图片/视频的保存路径
    img-size:网络输入图片大小
    conf-thres:置信度阈值
    iou-thres:做nms的iou阈值
    device:设置设备
    view-img:是否展示预测之后的图片/视频，默认False
    save-txt:是否将预测的框坐标以txt文件形式保存，默认False
    classes:设置只保留某一部分类别，形如0或者0 2 3
    agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
    augment:推理的时候进行多尺度，翻转等操作(TTA)推理
    update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='a_test/pic', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
```



如果需要生成xml文档

在utils/general.py添加

```python
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
```

> 不知道为啥，好像现在的yoloV5没有这个函数

然后将detect.py改成

```python
import argparse
import os
import shutil
import time
from pathlib import Path

import json

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    # 获取设置的参数数据
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    # 如果设备为GPU时， 使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model 确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier 设置第二次分类，默认不使用
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # 获取类别名字
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # 进行一次前向推理,测试程序是否正常
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # 输出json文件
    save_json = True
    content_json = []

    # path 图片/视频路径
    # img 进行resize+pad之后的图片
    # img0 原size图片
    # cap 当读取图片时为None，读取视频时为视频源
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        """
        前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
        h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[..., 0:4]为预测框坐标
        预测框坐标为xywh(中心点+宽长)格式
        pred[..., 4]为objectness置信度
        pred[..., 5:-1]为分类结果
        """
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        """
        pred:前向传播的输出
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别
        agnostic:进行nms是否也去除不同类别之间的框
        经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor]，长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        # 添加二次分类，默认不使用
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # 对每一张图片作处理
        for i, det in enumerate(pred):  # detections per image
            # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # 设置保存图片/视频的路径
            save_path = str(Path(out) / Path(p).name)
            # 设置保存框坐标txt文件的路径
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # 设置打印信息（图片长宽）
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            # 在原图上画框
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # 输出 json 文件
                    if save_json:
                        # windows下使用
                        file_name = save_path.split('\\')
                        # Linux下使用
                        # file_name = save_path.split('/')
                        content_dic = {
                            "name": file_name[len(file_name)-1],
                            "category": (names[int(cls)]),
                            "bbox": torch.tensor(xyxy).view(1, 4).view(-1).tolist(),
                            "score": conf.tolist()
                        }
                        content_json.append(content_dic)

            # Print time (inference + NMS)
            # 打印前向传播时间
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # 如果设置展示，则show图片/视频
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # 设置保存图片/视频
            # if save_img:
            #     if dataset.mode == 'images':
            #         cv2.imwrite(save_path, im0)
            #     else:
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #
            #             fourcc = 'mp4v'  # output video codec
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #         vid_writer.write(im0)

    if save_txt or save_img or save_json:
        print('Results saved to %s' % Path(out))
        # 将 json 数据写入文件
        with open(os.path.join(Path(out), 'result.json'), 'w') as f:
            json.dump(content_json, f)
    # 打印总时间
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    """
    weights:训练的权重
    source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    output:网络预测之后的图片/视频的保存路径
    img-size:网络输入图片大小
    conf-thres:置信度阈值
    iou-thres:做nms的iou阈值
    device:设置设备
    view-img:是否展示预测之后的图片/视频，默认False
    save-txt:是否将预测的框坐标以txt文件形式保存，默认False
    classes:设置只保留某一部分类别，形如0或者0 2 3
    agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
    augment:推理的时候进行多尺度，翻转等操作(TTA)推理
    update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../tile/testA_imgs', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1600, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='detect_img/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                # 去除pt文件中的优化器等信息
                strip_optimizer(opt.weights)
        else:
            detect()
```

更改完在输出目录会输出一个json文件

为了实现图片的传输，这边采用base64编码将图片和base64互相转化
~~~Python
# -*- coding: utf-8 -*-
  
import base64
import os
import json
import re
import xlrd, xlwt


def toBase64():
    insect = {
        'Drosicha corpulenta': "草履蚧 (Drosicha corpulenta (Kuwana))",
        'Erthesina fullo': "麻皮蝽 (Erthesina fullo (Thunberg))",
        'Anoplophora chinensis': "星天牛 (Anoplophora chinensis Forster,1771)",
        'Chalcophora japonica': "日本脊吉丁 (Chalcophora japonica(Gory))",
        'Apriona germar': "桑天牛 (Apriona germari(Hope))",
        'plagiodera versicolora': "柳蓝叶甲 (plagiodera versicolora )",
        'Monochamus alternatus Hope': "松墨天牛 (Monochamus alternatus Hope, 1842)",
        'Cnidocampa flavescens': "黄刺蛾 (Cnidocampa flavescens（Walker）)",
        'Latoria consocia Walker': "褐边绿刺蛾 (Latoria consocia Walker)",
        'Psilogramma menephron': "霜天蛾 (Psilogramma menephron(Gramer.))",
        'Hyphantria cunea': "美国白蛾 (Hyphantria cunea (Drury))",
        'Sericinus montelus Grey': "丝带凤蝶 (Sericinus montelus Grey)",
        'Spilarctia subcarnea': "人纹污灯蛾 (Spilarctia subcarnea (Walker))",
        'Micromelalopha troglodyta': "杨小舟蛾 (Micromelalopha troglodyta (Graeser))",
        'Clostera anachoreta': "杨扇舟蛾 (Clostera anachoreta (Denis et Schiffermüller, 1775))"
    }
    dir_path = os.path.dirname(os.path.abspath(__file__))
    json_path = dir_path + "\\detect_img\\output\\"
    with open(json_path + 'result.json', 'r') as f:
        load_dict = json.load(f)
        item_dir = load_dict[0]
        category = item_dir["name"]
        mesLs = getMessage(insect[category])
        allName = mesLs[0]
        chineseName = re.match(r'(.*?)\((.*?)\)(.*?)', allName).group(1).strip()
        englishName = allName.replace(chineseName, '')[1: -1]
        item_dir["name"] = chineseName
        item_dir["english"] = englishName
        item_dir["category"] = mesLs[1]
        item_dir["introduce"] = mesLs[2]

    print(item_dir)
    return item_dir


"""
def writeJson():

    insect = {
        'Drosicha corpulenta': "草履蚧 (Drosicha corpulenta (Kuwana))",
        'Erthesina fullo': "麻皮蝽 (Erthesina fullo (Thunberg))",
        'Anoplophora chinensis': "星天牛 (Anoplophora chinensis Forster,1771)",
        'Chalcophora japonica': "日本脊吉丁 (Chalcophora japonica(Gory))",
        'Apriona germar': "桑天牛 (Apriona germari(Hope))",
        'plagiodera versicolora': "柳蓝叶甲 (plagiodera versicolora )",
        'Monochamus alternatus Hope': "松墨天牛 (Monochamus alternatus Hope, 1842)",
        'Cnidocampa flavescens': "黄刺蛾 (Cnidocampa flavescens（Walker）)",
        'Latoria consocia Walker': "褐边绿刺蛾 (Latoria consocia Walker)",
        'Psilogramma menephron': "霜天蛾 (Psilogramma menephron(Gramer.))",
        'Hyphantria cunea': "美国白蛾 (Hyphantria cunea (Drury))",
        'Sericinus montelus Grey': "丝带凤蝶 (Sericinus montelus Grey)",
        'Spilarctia subcarnea': "人纹污灯蛾 (Spilarctia subcarnea (Walker))",
        'Micromelalopha troglodyta': "杨小舟蛾 (Micromelalopha troglodyta (Graeser))",
        'Clostera anachoreta': "杨扇舟蛾 (Clostera anachoreta (Denis et Schiffermüller, 1775))"
    }
    dir_path = os.path.dirname(os.path.abspath(__file__))
    json_path = dir_path + "\\detect_img\\output\\"
    with open(json_path + 'result.json', 'r') as f:
        load_dict = json.load(f)
        item_dir = load_dict[0]
        category = item_dir["name"]
        mesLs = getMessage(insect[category])
        item_dir["name"] = mesLs[0]
        item_dir["category"] = mesLs[1]
        item_dir["introduce"] = mesLs[2]

        print(mesLs)

    print("Return Success!")
    return item_dir
"""


def deBase64(base64_data):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    img_path = dir_path + "\\detect_img\\input\\"
    print("Load img...")
    imgData = base64.b64decode(base64_data)
    file = open(img_path + 'result.jpg', 'wb')
    file.write(imgData)
    file.close()
    print("Img has been loaded!")


# 从CSV中提取数据
def getMessage(data):
    dataXlsx = xlrd.open_workbook('pest.xls')
    table = dataXlsx.sheet_by_name(u'Sheet1')  # 通过名称获取
    for i in range(table.nrows):
        lsMs = table.row_values(i)
        if data == lsMs[0]:
            return table.row_values(i)


if __name__ == '__main__':
    toBase64()
~~~

然后在主函数中使用Flask接收信息，如果对方发来一个请求，则将base64转化为本地图片，detect对其进行检测，然后将输出的json信息发送回去

~~~Python
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

~~~