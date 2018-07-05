#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

capL = cv2.VideoCapture(2)
capR = cv2.VideoCapture(0)

imgL = np.zeros((480, 640, 3), np.uint8)
imgR = np.zeros((480, 640, 3), np.uint8)

stereo = None

opencv_measure_version = int(cv2.__version__.split('.')[0])
windowSize = 5
minDisp = 32
numDisp = 144 - minDisp

# for OpenCV3
stereo = cv2.StereoSGBM_create(
        minDisparity=minDisp,
        numDisparities=numDisp,
        blockSize=16,
        P1=8*3*windowSize**2,
        P2=32*3*windowSize**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
capL.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_WIDTH)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
capR.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_WIDTH)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

while True:
    capL.read(imgL)
    capR.read(imgR)

    # create gray images
    imgGrayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgGrayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # calculate histogram
    imtGrayL = cv2.equalizeHist(imgGrayL)
    imtGrayR = cv2.equalizeHist(imgGrayR)

    # through gausiann filter
    imgGrayL = cv2.GaussianBlur(imgGrayL, (5, 5), 0)
    imgGrayR = cv2.GaussianBlur(imgGrayR, (5, 5), 0)

    cv2.imshow("image left", imgGrayL)
    cv2.imshow("image right", imgGrayR)

    # calculate disparity
    disparity = stereo.compute(imgGrayL, imgGrayR).astype(np.float32)/16
    disparity = (disparity - minDisp) / numDisp

    cv2.imshow("disparity", disparity)

    k = cv2.waitKey(33)
    if k == ord('q'):
        break
