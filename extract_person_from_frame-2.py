from models import *
from utils import *

def verify_image(img_file):
     #test image
     try:
        v_image = Image.open(img_file)
        v_image.verify()
        return True;
        #is valid
        #print("valid file: "+img_file)
     except OSError:
        return False;

import os, sys, time, datetime, random
import torch
import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4
input_help = "Put path to image here"
parser = argparse.ArgumentParser(description='Tool for making FAQ from existing qa data')
parser.add_argument("--input", "-i", help=f"{input_help}")
parser.add_argument("--config_path", "-p", default='yolov3.cfg', help="Path to yolo3 config")
parser.add_argument("--weights_path", "-w", default = 'yolov3.weights', help="Path to yolo3 weights")
parser.add_argument("--class_path", "-c", default = 'coco.names', help = " Path to yolo3 classes")
parser.add_argument("--output", "-o", default='DataSet')
args = parser.parse_args()
print(args)


config_path=args.config_path
weights_path=args.weights_path
class_path=args.class_path
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

def get_data(path = "/content/oXfAHx-VuF0.jpg"):
    img_path = path
    out_path = "/content/drive/My Drive/"  + args.output
    #tmp_dir = 'tmp'+path.split('/')[-1]
    #os.mkdir(tmp_dir)
    prev_time = time.time()
    img = Image.open(img_path)
    img = img.transpose(Image.ROTATE_270)
    img.save("11111.jpg")
    img1 = img
    detections = detect_image(img)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print ('Inference Time: %s' % (inference_time))



    img = np.array(img)
    

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size * 1.2 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size * 1.2 / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is None:
      return None
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

    # browse detections and draw bounding boxes
        ctr = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print(x1)
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            if classes[int(cls_pred)] == 'person':
                out_image = img1.crop((x1.item(),y1.item(),x1.item()+box_w.item(), y1.item()+box_h.item()))
            
                #full.save(tmp_dir+'/full.jpg')
            

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)
    img1 = out_image
    boxes, _ = mtcnn.detect(img1)
    print('Boxes:', boxes)
    if boxes is None:
        return None
    x = boxes[0][0]
    y = boxes[0][1]
    w = abs(boxes[0][0]-boxes[0][2])
    h = abs(boxes[0][1]-boxes[0][3])
    x1 = int(x - x * 0.3) 
    y1 = int(y - 0.8 * y) 
    x2 = int(x+w + 0.1 * (x+w)) 
    y2 = int(y + h + 0.2 * (y + h))
    out_image_face = img1.crop(boxes[0])
    out_image = img1.crop((x1,y1,x2,y2))
    #print(out_path + path.split('/')[-1]+ 'head.jpg')
    full_head = out_image
    try:
        os.mkdir(out_path+'/HardHead')
        os.mkdir(out_path+'/HardHead/0')
    except:
        pass
    
    print(out_path + '/HardHead/0' + img_path.split('/')[-1])
    full_head.save(out_path + '/HardHead/0/' + img_path.split('/')[-1])
    print('===='*10)
    only_face = out_image_face
    try:
      os.mkdir(out_path+'/Glasses')
      os.mkdir(out_path+'/Glasses/0')
    except:
      pass
    only_face.save(out_path + '/Glasses/0/' + img_path.split('/')[-1])
    only_body = img1.crop((0, y2, img1.size[0], img1.size[1]))
    try:
      os.mkdir(out_path+'/Stuff')
      os.mkdir(out_path+'/Stuff/0')
    except:
      pass
    only_body.save(out_path + '/Stuff/0/' + img_path.split('/')[-1])
   
path = "/content/drive/My Drive/" + args.input

get_data(path)
