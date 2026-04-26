import os
import sys
import cv2
from kiwisolver import Constraint
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class Angle(tf.keras.constraints.Constraint):
    def call(self, w):
        w -= tf.floor((w+np.pi)/(np.pi*2))
        return w

class Distortion(tf.keras.layers.Layer):
    def __init__(self, H, W, **kwargs):
        super().__init__(**kwargs)
        self.H = H
        self.W = W
    def get_config(self):
        config = super().get_config()
        config.update({'H': self.H, 'W': self.W})
        return config
    def build(self, input_shape):
        W2 = self.W//2
        H2 = self.H//2
        nx,ny = np.meshgrid(np.arange(-W2,self.W-W2), np.arange(-H2, self.H-H2))
        self.nx = nx
        self.ny = ny

        self.a0 = self.add_weight(name='a0', shape=(2,), initializer='zeros')
        self.a1 = self.add_weight(name='a1', shape=(2,), initializer='zeros')
        self.theta = self.add_weight(name='theta', shape=(1,), initializer='zeros', constraint=Angle())
        gx = np.array([-1,0,1]).reshape((1,3,1,1))
        gy = np.array([-1,0,1]).reshape((3,1,1,1))
        self.gx = tf.constant(gx, dtype=tf.float32)
        self.gy = tf.constant(gy, dtype=tf.float32)
        pass    
    def call(self, inputs):
        ny2 = self.ny + self.a0[0] + self.a1[0]*self.ny
        nx2 = self.nx + self.a0[1] + self.a1[1]*self.nx
        ny3 = ny2*tf.cos(self.theta) + nx2*tf.sin(self.theta)
        nx3 = -ny2*tf.sin(self.theta) + nx2*tf.cos(self.theta)
        dx = nx3-self.nx
        dy = ny3-self.ny
        Ix = tf.nn.conv2d(inputs, self.gx,(1,1),'SAME')
        Iy = tf.nn.conv2d(inputs, self.gy,(1,1),'SAME')
        out = inputs + Ix*dx + Iy*dy
        return out

def create_model(H,W):
    inp = tf.keras.layers.Input(shape=(None,None,1))
    dist = Distortion(H=H, W=W)
    out = dist(inp)
    model = tf.keras.Model(inp, out)
    return model

def mse_loss(y_true, y_pred):
    border_pad = 5
    diff = y_true-y_pred
    loss_val = tf.reduce_mean(tf.square(diff[:,border_pad:-border_pad,border_pad:-border_pad,:]))
    return loss_val

def test_distortion():
    img_fname = r'C:\dev\repos\ImageDecomposition\src\main\resources\sem_images\SRAM_22nm_filtered.jpg'
    img = cv2.imread(img_fname, 0).astype(np.float32)
    img = img/255.0
    H,W = img.shape
    model = create_model(H,W)
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss=mse_loss)


    gt_img = img.copy()
    theta = 0.1*np.pi/180
    cv2.warpAffine(gt_img, np.array([[np.cos(theta),np.sin(theta),0.25],[-np.sin(theta),np.cos(theta),0.25]]), (W,H), gt_img)
    img4 = img[np.newaxis,:,:,np.newaxis]
    gt_img4 = gt_img[np.newaxis,:,:,np.newaxis]
    model.fit(img4, gt_img4, epochs=200)
    l =model.get_layer(name='distortion')
    print(l.get_weights())
    out = model.predict(img4)
    out = out[0,:,:,0]
    ax1 = plt.subplot(231)
    plt.imshow(gt_img)
    plt.subplot(232, sharex=ax1, sharey=ax1), plt.imshow(out)
    plt.subplot(233, sharex=ax1, sharey=ax1), plt.imshow(out-gt_img)
    plt.subplot(234, sharex=ax1, sharey=ax1), plt.imshow(img)
    plt.subplot(235, sharex=ax1, sharey=ax1), plt.imshow(img-gt_img)
    plt.show()
    pass


if __name__ == '__main__':
    test_distortion()
    pass