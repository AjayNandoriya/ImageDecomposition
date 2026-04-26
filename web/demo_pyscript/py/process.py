import os
import numpy as np
import cv2
import logging

LOGGER = logging.getLogger(__name__)
from matplotlib import pyplot as plt
import plotly.express as px

class MyApp(object):
    def __init__(self) -> None:
        self.ref_img = None
        self.test_img = None
        self.base_img = None
        self.diff_img = None 
        pass
    def run(self):
        self.diff_img = compare(self.ref_img, self.test_img)
    def plot(self):
        fig, axes = plt.subplots(2,2,sharex=True, sharey=True)
        if self.ref_img is not None: 
            axes[0,0].imshow(self.ref_img)
            axes[0,0].set_title('ref')
        if self.test_img is not None:
            axes[0,1].imshow(self.test_img)
            axes[0,1].set_title('test')
        if self.base_img is not None:
            axes[1,0].imshow(self.base_img)
            axes[1,0].set_title('base')
        if self.diff_img is not None:
            axes[1,1].imshow(self.diff_img)
            axes[1,1].set_title('diff')
        return fig
    
    def plot_plotly(self):
        imgs, titles = [],[]
        if self.ref_img is not None: 
            imgs.append(self.ref_img)
            titles.append('ref')
        if self.test_img is not None: 
            imgs.append(self.test_img)
            titles.append('test')
        if self.base_img is not None: 
            imgs.append(self.base_img)
            titles.append('base')
        if self.diff_img is not None: 
            imgs.append(self.diff_img)
            titles.append('diff')

        fig = px.imshow(np.array(imgs))
        # Set facet titles
        for i, sigma in enumerate(titles):
            fig.layout.annotations[i]['text'] = titles[i]
        return fig
    
def compare(img1:np.ndarray, img2:np.ndarray)->np.ndarray:
    LOGGER.info(f'got images with size {img1.shape} {img2.shape}')
    diff_img = img1.astype(np.float32) - img2.astype(np.float32)
    ret, th_img = cv2.threshold(diff_img,0,255, cv2.CV_8U)

    LOGGER.info(f'returning result image with {th_img.sum()} count.')
    return th_img

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pass