import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def get_avg_angular_kernel(kernel_half_width, angle, tight=True):
    """
    Get the average angular kernel.

    :param kernel_half_width: The half-width of the kernel.
    :param angle: The angle of the kernel.
    :return: The average angular kernel.
    """
    # Initialise the kernel.
    horizontal_kernel = np.zeros((2*kernel_half_width+1, 2*kernel_half_width+1))
    horizontal_kernel[kernel_half_width,:]=1
    # Calculate the kernel.
    rot_mat = cv2.getRotationMatrix2D((kernel_half_width, kernel_half_width), angle, 1.0)
    angular_kernel = cv2.warpAffine(horizontal_kernel, rot_mat, horizontal_kernel.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Normalise the kernel.
    angular_kernel = angular_kernel / np.sum(angular_kernel)

    if tight:
        x1 = int(np.floor(kernel_half_width -rot_mat[0,0]*kernel_half_width))
        x2 = int(np.ceil(kernel_half_width +rot_mat[0,0]*kernel_half_width))
        y1 = int(np.floor(kernel_half_width -rot_mat[0,1]*kernel_half_width))
        y2 = int(np.ceil(kernel_half_width +rot_mat[0,1]*kernel_half_width))
        angular_kernel = angular_kernel[y1:y2+1, x1:x2+1]
    # Return the kernel.
    return angular_kernel

def test_sem2gds():
    sem_fname = os.path.join(os.path.dirname(__file__), '..','resources','sem_images','SRAM_22nm.jpg')
    gds_fname = os.path.join(os.path.dirname(__file__), '..','resources','sem_images','gds.png')
    sem_img = cv2.imread(sem_fname, cv2.IMREAD_GRAYSCALE)
    gds_img = sem2gds(sem_img)
    cv2.imwrite(gds_fname, gds_img)
    pass

def sem2gds(sem_img):
    fimg = cv2.fastNlMeansDenoising(sem_img, None, 20, 7, 21)
    # fimg = cv2.morphologyEx(fimg, cv2.MORPH_TOPHAT, np.ones((9,9),np.uint8))
    bimg = (fimg<85).astype(np.uint8)*255

    dist = cv2.distanceTransform(bimg, cv2.DIST_L2, 3)
    ret, dist1 = cv2.threshold(dist, 0.3*dist.max(), 255, 0)
    kernel = np.ones((7,7),np.uint8)
    dist1 = cv2.morphologyEx(dist1.astype(np.uint8),cv2.MORPH_DILATE, kernel, iterations = 2)

    markers = np.zeros(dist.shape, dtype=np.int32)
    dist_8u = dist1.astype('uint8')
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)
    markers = cv2.watershed(cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR), markers)

    bimg = (markers!=10) & (fimg<105)
    bimg = bimg.astype(np.uint8)*255
    bimg = cv2.morphologyEx(bimg, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
    
    # bimg = cv2.GaussianBlur(bimg, (5,5), 0)
    if 0:
        ax1=plt.subplot(221)
        ax1.imshow(sem_img, cmap='gray')
        ax2=plt.subplot(222, sharex=ax1, sharey=ax1)
        ax2.imshow(fimg, cmap='gray')
        ax2=plt.subplot(223, sharex=ax1, sharey=ax1)
        ax2.imshow(bimg, cmap='gray')
        plt.show()
    return bimg

if __name__ == '__main__':
    test_sem2gds()