import os

import cv2
import numpy as np 
import pandas as pd
import statistics as sts

from skimage import img_as_ubyte
from skimage.transform import rotate
from tqdm import tqdm

def detect(cascade, img):
    return cascade.detectMultiScale(
        img,
        scaleFactor=1.05,
        minNeighbors=3,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
SCALE_FACTOR = 6
# bulk part of dataset has images AMI Ear Dataset, where images has size 492x700x3
DEFAULT_WIDTH = 492 // SCALE_FACTOR
DEFAULT_HEIGHT = 700 // SCALE_FACTOR
# location of OpenCV Haar Cascade Classifiers:
base_cascade_dir = './cascades/'

# xml files describing our haar cascade classifiers
l_cascade_path = base_cascade_dir + 'haarcascade_mcs_leftear.xml'
r_cascade_path = base_cascade_dir + 'haarcascade_mcs_rightear.xml'

# build cv2 Cascade Classifiers
left_cascade = cv2.CascadeClassifier(l_cascade_path)
right_cascade = cv2.CascadeClassifier(r_cascade_path)

TRAIN_DATADIR = "../ds_train"
TRAIN_RES_DATADIR = '../processed_focused_dataset/train_classes'

VALID_DATADIR = "../ds_valid"
VALID_RES_DATADIR = '../processed_focused_dataset/valid_classes'

UNK_DATADIR = '../ds_unknown'
UNK_RES_DATADIR = '../processed_focused_dataset/train_classes'

def prepare_dataset(datadir, res_datadir, one_class=None, one_class_limit=50):
    SUBSETS = [os.path.join(subset) for subset in os.listdir(datadir) 
                        if os.path.isdir(os.path.join(datadir, subset))]
    labels_stats = {}
    l_ears_stats = {'w': [], 'h': []}
    r_ears_stats = {'w': [], 'h': []}
    total_imgs = 0
    for _, subset in enumerate(SUBSETS):
        subset_path = os.path.join(datadir, subset)
        imgs = os.listdir(subset_path)
        loaded_img_counters = {}
        for img_i, img_name in enumerate(tqdm(imgs, desc=f'Detecting ears with Haar Cascades in {subset_path}...')):
            try: 
                gray = cv2.imread(os.path.join(subset_path, img_name), cv2.IMREAD_GRAYSCALE)
                if one_class:
                    # we dont want too smal pics
                    if gray.shape[0] < 50:
                        continue
                    # and we dont want too much pics from one folder
                    # if img_i > one_class_limit:
                    #     break
                resized_gray = cv2.resize(gray, (DEFAULT_WIDTH, DEFAULT_HEIGHT))
                total_imgs += 1
                if one_class:
                    img_pref = str(one_class)
                else:
                    img_pref = int(float(img_name[:3]))
                if not img_pref in labels_stats:
                    labels_stats[img_pref] = {'label': img_pref, 'imgs_num': 0,'left_ears': 0, 'right_ears': 0}
                labels_stats[img_pref]['imgs_num'] += 1
                for angle in range(180):
                    rotated = rotate(resized_gray, angle)
                    left_ears = detect(left_cascade, img_as_ubyte(rotated))
                    right_ears = detect(right_cascade, img_as_ubyte(rotated))
                    if len(left_ears)>0 or len(right_ears)>0:
                        break
                if len(left_ears)>0:
                    labels_stats[img_pref]['left_ears'] += 1
                    ear_dir = os.path.join(res_datadir, str(img_pref))
                    if not os.path.exists(ear_dir):
                        os.mkdir(ear_dir)
                    ear_dir = os.path.join(ear_dir, 'haar_detect')
                    if not os.path.exists(ear_dir):
                        os.mkdir(ear_dir)
                    ear_dir = os.path.join(ear_dir, 'left_ear')
                    if not os.path.exists(ear_dir):
                        os.mkdir(ear_dir)
                    ear_path = os.path.join(ear_dir, img_name)
                    im = cv2.imread(os.path.join(subset_path, img_name))
                    focused_ear = np.zeros((im.shape[0], im.shape[1]))
                    focused_ear = im[:,:,2].astype(int) - im[:,:,1]//0.7 # 0.7 found as best coef to get more visible ear shape
                    focused_ear[focused_ear < 0] = 0
                    cv2.imwrite(ear_path, focused_ear)
                    l_ears_stats['w'].append(resized_gray.shape[0])
                    l_ears_stats['h'].append(resized_gray.shape[1])


                if len(right_ears)>0:
                    labels_stats[img_pref]['right_ears'] += 1
                    ear_dir = os.path.join(res_datadir, str(img_pref))
                    if not os.path.exists(ear_dir):
                        os.mkdir(ear_dir)
                    ear_dir = os.path.join(ear_dir, 'haar_detect')
                    if not os.path.exists(ear_dir):
                        os.mkdir(ear_dir)
                    ear_dir = os.path.join(ear_dir, 'right_ear')
                    if not os.path.exists(ear_dir):
                        os.mkdir(ear_dir)
                    ear_path = os.path.join(ear_dir, img_name)
                    im = cv2.imread(os.path.join(subset_path, img_name))
                    focused_ear = np.zeros((im.shape[0], im.shape[1]))
                    focused_ear = im[:,:,2].astype(int) - im[:,:,1]//0.7 # 0.7 found as best coef to get more visible ear shape
                    focused_ear[focused_ear < 0] = 0
                    blur = cv2.blur(focused_ear, (5, 5))  # With kernel size depending upon image size
                    # cv2.imwrite('graaaay.png',blur)
                    if np.average(blur) < 3:
                        # print('0.0')
                        continue
                    # else:
                        # print('+.+')
                    cv2.imwrite(ear_path, focused_ear)
                    r_ears_stats['w'].append(resized_gray.shape[0])
                    r_ears_stats['h'].append(resized_gray.shape[1])
        
            except Exception as e:
                print(f'Error {e}; image {os.path.join(subset_path, img_name)}')

    per_df = pd.DataFrame.from_records([row for row in labels_stats.values()])
    per_df.to_csv('another_dataset_stats.csv')
    r_ear_stat_df = pd.DataFrame(r_ears_stats)
    l_ear_stat_df = pd.DataFrame(l_ears_stats)
    print(f'Total ears detected: {len(r_ear_stat_df) + len(l_ear_stat_df)}, total images: {total_imgs}')
    # rec_res_h = int(sts.mean([r_ear_stat_df.h.quantile(q=0.5), l_ear_stat_df.h.quantile(q=0.5)]))
    # rec_res_w = int(sts.mean([r_ear_stat_df.w.quantile(q=0.5), l_ear_stat_df.w.quantile(q=0.5)]))
    # print(f'Recomended resize dimentions for further use (WIDTH, HEIGHT): {rec_res_w, rec_res_h}')
    print(f'Dataset processed and saved at {res_datadir}.')


prepare_dataset(TRAIN_DATADIR, TRAIN_RES_DATADIR)
prepare_dataset(VALID_DATADIR, VALID_RES_DATADIR)
prepare_dataset(UNK_DATADIR, UNK_RES_DATADIR, one_class='unknown')
