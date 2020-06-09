import sys
import json
from tensorflow.keras import models

import numpy as np
import cv2

from train import f1_m

IMG_WIDTH = 48
IMG_HEIGHT = 80


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_pathes = sys.argv[1].split(',')
    else:
        print('Running quick demo...')
        print('Hint: to make predictions on your images pass list to the script like this:')
        print("python3 predict.py './image1.png', './image2.jpg'")
        img_pathes = ['./samples/test_0001.png', './samples/001 (13).jpg', './samples/000_down_ear.jpg', './samples/000_front_ear.jpg']
    with open('labels_dict.json', 'r') as f:
        labels_dict = json.load(f)
    model = models.load_model('lew_cnn.h5', custom_objects={'f1_m': f1_m})
    for img_path in img_pathes:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        x = img.reshape(-1, *img.shape)
        pred_y = model.predict(x)
        breakpoint()
        pred_y_i = pred_y.argmax(axis=1)[0]
        for pred_y_i in pred_y_is:
            label = labels_dict[str(pred_y_i)]
            print(f'Predicted label: {label}')