import os
import json

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from skimage import img_as_ubyte
from tqdm import tqdm

from transformations import anticlockwise_rotation, clockwise_rotation, h_flip,  v_flip, warp_shift, add_noise, blur_image, change_brightness

TRANSFORMATIONS = [
                    anticlockwise_rotation, 
                    clockwise_rotation, 
                    v_flip, 
                    h_flip,
                    blur_image,
                    change_brightness
                  ]

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def main():
    DATADIR = '../processed_focused_dataset/train_classes'
    LABELS = []
    IMG_WIDTH = 48
    IMG_HEIGHT = 80
    for cl in os.listdir(DATADIR):
        if os.path.isdir(os.path.join(DATADIR, cl)):
            if os.path.exists(os.path.join(DATADIR, cl)):
                LABELS.append(os.path.join(cl))
    assert len(LABELS) > 0
    labels_dict = {i:label for i, label in enumerate(sorted(LABELS))}
    with open('labels_dict.json', 'w') as f:
        json.dump(labels_dict, f)
    keys=[*labels_dict.keys()]
    values=[*labels_dict.values()]
    unknown_index = keys[values.index('unknown')]
    X = []
    y = []

    # sort class folders to get unknown on the last position, reason: to have easier access to unknown
    for label_i, label in enumerate(sorted(LABELS)):
        path = os.path.join(DATADIR, label, 'haar_detect')
        for side in os.listdir(path):
            side_path = os.path.join(path, side)
            if os.path.isdir(side_path):
                for img in os.listdir(side_path):
                    try:
                        gray_img = cv2.imread(os.path.join(side_path, img))
                        resized_img = cv2.resize(gray_img, (IMG_WIDTH, IMG_HEIGHT))
                        X.append(resized_img)
                        y.append(label_i)
                        for transformation in TRANSFORMATIONS:
                            transformed_image = transformation(resized_img)
                            transformed_image = img_as_ubyte(transformed_image)
                            X.append(transformed_image)
                            y.append(label_i)
                    except Exception as e:
                        pass
    y = to_categorical(y)
    X_train = np.asanyarray(X).reshape(-1, *X[0].shape)
    y_train = y
    # image normalization
    X_train = X_train/255.0

    VALID_DATADIR = '../processed_focused_dataset/valid_classes'
    VALID_LABELS = []
    X_test = []
    y_test = []
    IMG_WIDTH = 48
    IMG_HEIGHT = 80
    for cl in os.listdir(VALID_DATADIR):
        if os.path.isdir(os.path.join(VALID_DATADIR, cl)):
            if os.path.exists(os.path.join(VALID_DATADIR, cl, 'haar_detect')):
                VALID_LABELS.append(os.path.join(cl, 'haar_detect'))
    assert len(VALID_LABELS) > 0
    unknown_bin_vec = np.zeros(y.shape[1])
    unknown_bin_vec[unknown_index] = 1

    for _, label in enumerate(VALID_LABELS):
        path = os.path.join(VALID_DATADIR, label)
        for side in os.listdir(path):
            side_path = os.path.join(path, side)
            if os.path.isdir(side_path):
                for img in os.listdir(side_path):
                    try:
                        gray_img = cv2.imread(os.path.join(side_path, img))
                        resized_img = cv2.resize(gray_img, (IMG_WIDTH, IMG_HEIGHT)) 
                        X_test.append(resized_img)
                        y_test.append(unknown_bin_vec)
                    except Exception as e:
                        pass
    y_test = np.asanyarray(y_test)
    X_test = np.asanyarray(X_test).reshape(-1, *X_test[0].shape)
    # image normalization
    X_test = X_test/255.0

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam', metrics=['accuracy', f1_m])

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

    # evaluate the model
    # INFO: keras has problems with displaying metrics during fit run, so we evaluate 
    # model on each subset to get real scores.
    scores  = model.evaluate(X_train, y_train)
    print('----- Validation scores: -----')
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])
    print('F1: %.3f' % scores[2])

    scores  = model.evaluate(X_test, y_test)
    print('----- Validation scores: -----')
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])
    print('F1: %.3f' % scores[2])

    # save confidence matricies on train/val subsets
    y_pred = model.predict(X_test)
    con_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    np.savetxt('test_conf_mat.txt', con_mat, delimiter=',', fmt='%i')

    pred_y = model.predict(X_train)
    con_mat = confusion_matrix(y_train.argmax(axis=1), pred_y.argmax(axis=1))
    np.savetxt('train_conf_mat.txt', con_mat, delimiter=',', fmt='%i')

    # save the model
    model.save('lew_cnn.h5')

if __name__ == '__main__':
    main()