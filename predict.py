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
        img_pathes = ['samples/002_left_ear.jpg', 
                      'samples/021 (12).jpg', 
                      'samples/021 (15).jpg', 
                      'samples/042_right_ear.jpg',
                      'samples/078_down_ear.jpg', 
                      'samples/084_right_ear.jpg', 
                      'samples/201.png', 
                      'samples/1130_contrast.png',
                      ]
    with open('labels_dict.json', 'r') as f:
        labels_dict = json.load(f)
    model = models.load_model('lew_cnn.h5', custom_objects={'f1_m': f1_m})
    for img_path in img_pathes:
        img = cv2.imread(img_path)
        if len(img.shape) < 3:
            print('WARNING! Only 3 channel images are allowed. Grayscale images are not supported!')
            raise Exception

        focused_ear = np.zeros((IMG_WIDTH, IMG_WIDTH))
        focused_ear = img[:,:,2].astype(int) - img[:,:,1]//0.7 # 0.7 found as best coef to get more visible ear shape
        focused_ear[focused_ear < 0] = 0
        focused_ear = focused_ear.astype('uint8')
        focused_ear = cv2.cvtColor(focused_ear, cv2.COLOR_GRAY2RGB)
        focused_ear = cv2.resize(focused_ear, (IMG_WIDTH, IMG_HEIGHT))
        norm_img = focused_ear/255.0
        x = norm_img.reshape(-1, *focused_ear.shape)
        pred_y = model.predict(x)
        pred_y_i = pred_y.argmax(axis=1)[0]
        label = labels_dict[str(pred_y_i)]
        print(f'img: {img_path}; predicted label: {label}')