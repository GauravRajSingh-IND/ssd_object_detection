import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from Module_13.main import classes

FONTFACE = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

def detect_object(net, image):

    # preprocessing steps - creating blob from image
    dim = (300, 300)
    mean = (127.5, 127.5, 127.5)
    blob = cv.dnn.blobFromImage(image, 1.0/127.5, size=dim, mean=mean, swapRB=True)

    # forward pass - blob object to network.
    net.setInput(blob)

    # prediction
    objects = net.forward()
    return objects

def display_text(image, text, x, y):

    # get text size.
    text_size = cv.getTextSize(text, FONT_SCALE, FONT_SCALE, THICKNESS)
    dim = text_size[0]
    baseline = text_size[1]

    # create a rectangle box.
    cv.rectangle(image, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv.FILLED)
    cv.putText(image, text, (x, y-5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv.LINE_AA)

def display_object(image, objects, threshold = 0.25):

    rows = image.shape[0]
    cols = image.shape[1]

    # loop over each object detected
    for num in range(objects.shape[2]):

        # find class and confidence.
        class_num = int(objects[0, 0, num, 1])
        confidence = float(objects[0, 0, num, 2])

        # get original coordinates from normalised coordinates
        x = int(objects[0, 0, num, 3]) * cols
        y = int(objects[0, 0, num, 4]) * rows
        w = int(objects[0, 0, num, 5] * cols - x)
        h = int(objects[0, 0, num, 6] * rows - y)

        # check if the object detected is of good quality.
        if confidence >= threshold:

            display_text(image, classes[class_num], x, y)
            cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

        plt.imshow(image[:,:,::-1])
        plt.show()

model_file = "frozen_inference_graph.pb"
config_file = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classes_file = "coco_class_labels.txt"

# read tensorflow model
model = cv.dnn.readNet(model=model_file, config=config_file)

# check class labels and store classes in classes variable
with open(classes_file, 'r') as file:
    classes = file.read().splitlines()

