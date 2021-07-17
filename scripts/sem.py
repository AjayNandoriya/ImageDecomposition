import os
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

def cmae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true-y_pred)[:,15:-15,15:-15,:])

def create_model(N_scale=4, N_features=16):
    kernel_size = [2**N_scale-1, 2**N_scale-1] 
    strides = (2**(N_scale-1), 2**(N_scale-1))
    inp = tf.keras.layers.Input(shape=(None,None,1))
    sections = tf.keras.layers.Conv2D(N_features, kernel_size=kernel_size, strides=strides, padding='same', activation='softmax')(inp)
    sections = tf.keras.layers.UpSampling2D(size=strides,interpolation='bilinear')(sections)
    features = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', activation='relu')(sections)
    out = tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation=None, name='weight_sum')(features)
    model = tf.keras.models.Model(inputs=[inp],outputs=[out])
    model.compile(loss=cmae, metrics='mse', loss_weights=10)
    return model

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_fname):
        img = cv2.imread(img_fname,0).astype(float)/255.0
        H,W = img.shape
        self.img4d = img.reshape((1,H,W,1))
        self.img4d = np.tile(self.img4d, (4,1,1,1))
    def __len__(self):
        return 10
    def __getitem__(self,i):
        return self.img4d, self.img4d


def create_base_model(N_scale=4):
    kernel_size=[2**N_scale, 2**N_scale]
    inp = tf.keras.layers.Input(shape=(None,None,1))
    dout = tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=kernel_size, padding='same')(inp)
    out = tf.keras.layers.UpSampling2D(size=kernel_size,interpolation='bilinear')(dout)
    model = tf.keras.models.Model(inputs=[inp],outputs=[out])
    model.compile(loss=cmae, metrics='mse')
    return model

def train():
    N_features = 4
    model_fname = os.path.join(os.path.dirname(__file__), 'sem_model_4.h5')
    img_fname = os.path.join(os.path.dirname(__file__),'..','data','sem_images','SRAM_22nm.jpg')
    dg = DataGenerator(img_fname)

    model = create_model(N_features=N_features)

    base_model = create_base_model()
    if os.path.isfile(model_fname):
        model.load_weights(model_fname)

    model.fit(dg,epochs=3, verbose=1)
    model.save(model_fname)

    img = dg.img4d[0,:,:,0]
    gt_img = img
    
    # verify the model
    out_img4d = model.predict(dg.img4d)
    out_img = out_img4d[0,:,:,0]

    diff = out_img-gt_img
    mae = np.mean(np.abs(diff[15:-15,15:-15]))
    print(mae)

    # base line
    base_out_img4d = base_model.predict(dg.img4d)
    base_out_img = base_out_img4d[0,:,:,0]

    base_diff = base_out_img-gt_img
    base_mae = np.mean(np.abs(base_diff[15:-15,15:-15]))
    print(base_mae)

    ax1=plt.subplot(231)
    plt.imshow(img)
    plt.subplot(232, sharex=ax1, sharey=ax1),plt.imshow(gt_img)
    plt.subplot(233, sharex=ax1, sharey=ax1),plt.imshow(out_img)
    plt.subplot(234, sharex=ax1, sharey=ax1),plt.imshow(diff),plt.title(f'{mae:0.04f}')
    plt.subplot(235, sharex=ax1, sharey=ax1),plt.imshow(base_out_img)
    plt.subplot(236, sharex=ax1, sharey=ax1),plt.imshow(base_diff),plt.title(f'{base_mae:0.04f}')


    # analyse the model
    features = model.get_layer('weight_sum').input
    debug_model = tf.keras.models.Model(inputs=[model.input], outputs=[features, model.output])
    features, out_img = debug_model.predict(dg.img4d)
    print(features.shape)
    plt.figure(2)
    for i in range(N_features):
        plt.subplot(4,4,i+1)
        plt.imshow(features[0,:,:,i])

    plt.show()

if __name__ == '__main__':
    train()