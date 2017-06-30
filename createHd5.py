import numpy as np
import os
import h5py
import cv2
import random
from scipy import misc

imageSizeInit = 128

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def startMaking(dirPath):
    path = '/home/strange/ML/pytorch-LapSRN/data/train2.h5'
    h5pData = h5py.File(path, "w")
    images = os.listdir(dirPath)
    count= 0
    data = np.zeros((32, 32, 1, 32*len(images)))
    label_x2 = np.zeros((64, 64, 1, 32*len(images)))
    label_x4 = np.zeros((128, 128, 1, 32*len(images)))
    for img in images:

        imgLoad = cv2.imread(os.path.join(dirPath,img))
        imgYCC = cv2.cvtColor(imgLoad, cv2.COLOR_BGR2YCR_CB)
        imageDouble = imgYCC[:, :, 0].astype(float)
        print img
        h, w = imageDouble.shape[:2]
        if ((h <=imageSizeInit) or (w <= imageSizeInit)):
            imageDouble = misc.imresize(imageDouble,(200, 200), interp='bicubic', mode=None)
            h, w = imageDouble.shape[:2]

        imgData = misc.imresize(imageDouble, 0.25, interp='bicubic', mode=None)
        img64 = misc.imresize(imageDouble, 0.5, interp='bicubic', mode=None)

        img64 = misc.imresize(img64, (h, w), interp='bicubic', mode=None)
        imgData = misc.imresize(imgData, (h, w), interp='bicubic', mode=None)

        imageDouble = im2double(imageDouble)
        img64 = im2double(img64)
        imgData = im2double(imgData)

        tiles =32

        for i in range(1, tiles + 1):
            x = random.randint(0, w - imageSizeInit - 1)
            y = random.randint(0, h - imageSizeInit - 1)

            label_x4[:, :, 0, count] = misc.imresize(imageDouble[y:y + imageSizeInit, x:x + imageSizeInit],(128,128),interp='bicubic',mode=None)
            label_x2[:, :, 0, count] = misc.imresize(img64[y:y + imageSizeInit, x:x + imageSizeInit],(64,64),interp='bicubic',mode=None)
            data[:, :, 0, count] = misc.imresize(imgData[y:y + imageSizeInit, x:x + imageSizeInit],(32,32),interp='bicubic',mode=None)

            count += 1

    data = np.transpose(data, axes=(3,2,0,1))
    label_x4 = np.transpose(label_x4, axes=(3,2,0,1))
    label_x2 = np.transpose(label_x2, axes=(3,2,0,1))

    h5pData['data']        = data
    h5pData['label_x4']    = label_x4
    h5pData['label_x2']    = label_x2
    h5pData.close()

pathImages = '/home/strange/ML/pytorch-LapSRN/testLapSRN'
startMaking(pathImages)