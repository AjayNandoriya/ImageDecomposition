import os
import sys
import cv2
from matplotlib import pyplot as plt
MAIN_PATH = os.path.join(os.path.dirname(__file__),'..','python')
sys.path.append(MAIN_PATH)

import image_kernels

def try_avg_kernel():
    img_fname = os.path.join(os.path.dirname(__file__),'..','resources','sem_images','SRAM_22nm.jpg')
    img = cv2.imread(img_fname,0)

    angle = 45
    kernel_half_width = 5
    kernel = image_kernels.get_avg_angular_kernel(kernel_half_width, angle)

    fimg = cv2.filter2D(img,-1,kernel)
    ax1=plt.subplot(121)
    plt.imshow(img,cmap = 'gray')
    plt.title('Original')
    plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.imshow(fimg,cmap = 'gray')
    plt.title('Average')
    plt.show()

if __name__ == '__main__':
    try_avg_kernel()
