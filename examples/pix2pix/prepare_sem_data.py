import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

BASE_PATH = r'C:\Users\dext\Documents\Personal\repos\ImageDecomposition'

def convert2pix2pix_data():
    in_img_fname1 = BASE_PATH + r'\src\main\resources\sem_images\SRAM_22nm_gds.png'
    in_img_fname2 = BASE_PATH + r'\src\main\resources\sem_images\SRAM_22nm.jpg'

    img1 = cv2.imread(in_img_fname1, 0)
    img2 = cv2.imread(in_img_fname2, 0)


    img3 = cv2.fastNlMeansDenoising(img2, h=25, templateWindowSize=7, searchWindowSize=21)
    diff = img2.astype(np.float32) - img3.astype(np.float32)
    fig, axes = plt.subplots(1,3,sharex=True, sharey=True)
    axes[0].imshow(img2)
    axes[1].imshow(img3)
    axes[2].imshow(diff)
    plt.show()

    img = np.concatenate([img3, img1], axis=1)


    plt.imshow(img)
    plt.show()
    out_img_fname = r'C:\Users\dext\Documents\Personal\repos\ImageDecomposition\examples\pix2pix\data\1.jpg'
    cv2.imwrite(out_img_fname, img)


# write a function to add two number

if __name__ == '__main__':
    convert2pix2pix_data()
    pass