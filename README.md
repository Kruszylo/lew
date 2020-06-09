# LEW

## Intro

Labeled Ears in the Wild (Inspired by LFW (Labeled Face in the Wild), a public benchmark for face verification). 
Ears validation problem was succesfuly solved with `CNN` model:

```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 78, 46, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 39, 23, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 37, 21, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 18, 10, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 8, 64)         36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 4, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                131136    
_________________________________________________________________
dense_1 (Dense)              (None, 51)                3315      
=================================================================
Total params: 190,771
Trainable params: 190,771
Non-trainable params: 0
_________________________________________________________________
```

## Setup

Make sure you have Python 3.8 or later.

```bash
$ python3 -V
Python 3.8.1
```

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Dataset

My current dataset is an agregate of:

* [AMI Ear Database](http://ctim.ulpgc.es/research_works/ami_ear_database/)
* data from [Hitesh-Valecha/Ear_Biometric_System](https://github.com/Hitesh-Valecha/Ear_Biometric_System) project
* [EarVN1.0](https://data.mendeley.com/datasets/yws3v3mwx3/3)
* and crawled images from Google

The one on which model was trained you can find [here](https://drive.google.com/drive/folders/1K_n_zNj7RtztijhiXKZ41kVMW8oky-44?usp=sharing).
Most training images are in grayscale (one color channel), with size 82 × 116 pixels.

**Trainng subset** contains 51 classes:

* 50 classes of defined users
* 1 class for unknown users

Each class contains images of left and right ears. Normaly there is 6 images of left ear and 1 image of right ear.
Total number of images: 2382

**Validation subset** contains images of another 50 users which should not be verified among with other sample images of ears.

## Running

### Train

Every image is scaled to size 48 × 80. For maining more data augmentation is aplied, from each input image we create more transformed images:

* rotated anticlockwise on agnle 0-180
* rotated clockwise on angle 0-180
* flipped verticaly 
* flipped horizontaly
* blured with Gaussian filter
* changed brightness level

Use `train.py` for trainig new model.
Trained model will be saved in `.h5` file.

### Predict

Use `predict.py` for getting model predictions. There is a checkpoint `lewMobileNet.h5` of model trained 20 epochs with batch size 32, the checkpoint stats are:

||train|validation|
|---|---|---|
|acc|0.949|0.928|
|f1|0.949|0.929|

### Prepare dataset

I used nested convention for keeping dataset:

```bash
dataset_root
|_ classes
   |_ class1
   |  |_ haar_detect
   |     |_ left_ear
   |     |  |_ img1.png
   |     |  |_ img2.png
   |     |  ...
   |     |_ right_ear
   |        |_ img1.png
   |        |_ img2.png
   |        ...
   ...
```

In "wild" conditions other objects are captured on image among with ear, these objects is noise for our model, it's possible that there is no single ear on an image at all. Also, I wanted to protect the model from malicious attacks, namely make verification robust against objects which seems like ear (mushrooms, shells etc.). Thus as cleaning method I decided to use Haar Cascades.
Though Haar Cascades succesfuly find ears on image, sometimes it hard for them to find ear which is captured in not vertical position. Sometimes head is leaned forward or backward. It's seems natural for me, so to improve ear detection we rotate input image clockwise by 1 degree and checking if Haar Cascades can detect any ears on the image.
If right or left ear is detected, input image is saved in a grayscale to appropriate folder. I wanted to save cutted ear from the image, but I found that Haar Cascades used to detect only a part of ears.

### Rejeceted

As an idea of more secure pipeline I wanted to use Haar Cascades for ears as first step of validation.
With time I realized that model should deal with incorrect images if it was trained on proper data, also cascades tended to detect only part of the ear, which looks like problem of [dealing with high resolution and luck of negative samples](https://stackoverflow.com/questions/27196407/bad-trained-cascade-in-opencv). However, Haar Cascades are still powerfull enough for preparing better dataset.
