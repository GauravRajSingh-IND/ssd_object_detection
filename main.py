import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_object(net, image):

    # preprocessing steps - creating blob from image
    dim = (300, 300)
    mean = (127.5, 127.5, 127.5)
    blob = cv.dnn.blobFromImage(image, size=dim, mean=mean, swapRB=True)

    # forward pass - blob object to network.
    net.setInput(blob)

    # prediction
    objects = net.forward()
    return objects


model_file = "frozen_inference_graph.pb"
config_file = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classes_file = "coco_class_labels.txt"

# read tensorflow model
model = cv.dnn.readNet(model=model_file, config=config_file)

# check class labels and store classes in classes variable
with open(classes_file, 'r') as file:
    classes = file.read().splitlines()

