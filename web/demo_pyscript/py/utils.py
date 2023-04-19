import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def compare(img1:np.nparray, img2:np.ndarray)->np.ndarray:
    diff_img = img1.astype(np.float32) - img2.astype(np.float32)
    th_img = cv2.threshold(diff_img,0,255, cv2.CV_8U)
    return th_img