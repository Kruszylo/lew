import cv2
import numpy as np
import random

from skimage.util import random_noise
from skimage.transform import rotate, AffineTransform, warp


def anticlockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (3,3),0)

def change_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = np.random.uniform(0.5, 1.2)
    hsv[...,2] = hsv[...,2]*value
    _, _, img = cv2.split(hsv)
    return img

# I dont recommend warp_shifting, because it distorts image, but can be used in many use case like 
# classifying blur and non-blur images
def warp_shift(image): 
    transform = AffineTransform(translation=(0,40))  #chose x,y values according to your convinience
    warp_image = warp(image, transform, mode="wrap")
    return warp_image