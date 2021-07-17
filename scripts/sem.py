import os
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

def cmae(y_true, y_pred):
    diff = tf.square(y_true-y_pred)[:,15:-15,15:-15,:]
    full_diff = tf.reduce_mean(diff)
    clipped_diff = tf.clip_by_value(diff, clip_value_min=0.064, clip_value_max=100.0)
    clipped_loss = tf.reduce_mean(clipped_diff)
    loss = full_diff + clipped_loss*10
    return loss

def create_model(N_scale=4, N_features=16):
    kernel_size = [2**N_scale-1, 2**N_scale-1] 
    strides = (2**(N_scale), 2**(N_scale))
    inp = tf.keras.layers.Input(shape=(None,None,1))
    sections = tf.keras.layers.Conv2D(N_features, kernel_size=kernel_size, strides=strides, padding='same', activation='softmax', name='regions')(inp)
    sections = tf.keras.layers.UpSampling2D(size=strides,interpolation='bilinear')(sections)
    features = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', activation='tanh')(sections)
    out = tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation=None, name='weight_sum')(features)

    base_kernel_size = [2**(N_scale+1), 2**(N_scale+1)] 
    dout = tf.keras.layers.AveragePooling2D(pool_size=base_kernel_size, strides=base_kernel_size, padding='same')(inp)
    base_out = tf.keras.layers.UpSampling2D(size=base_kernel_size,interpolation='bilinear')(dout)
    out = tf.keras.layers.Add()([out, base_out])
    model = tf.keras.models.Model(inputs=[inp],outputs=[out])
    model.compile(loss=cmae, metrics='mse', loss_weights=50)
    return model

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_fname):
        img = cv2.imread(img_fname,0).astype(float)/255.0
        H,W = img.shape
        self.img4d = img.reshape((1,H,W,1))
        self.img4d = np.tile(self.img4d, (1,1,1,1))
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

def create_polynomial_model(N_scale=4, N_power=4):
    if N_power<=1:
        N_features = 1
    else:
        N_features = N_power**2
    kernel_size = [2**N_scale-1, 2**N_scale-1] 
    strides = (2**(N_scale-1), 2**(N_scale-1))
    inp = tf.keras.layers.Input(shape=(None,None,1))
    sections = tf.keras.layers.Conv2D(N_features, kernel_size=kernel_size, strides=strides, padding='same', name='scale_features')(inp)
    sections = tf.keras.layers.UpSampling2D(size=strides,interpolation='bilinear', name='features')(sections)
    # features = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', name='gen_features')(sections)
    out = tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation=None, name='weight_sum')(sections)

    model = tf.keras.models.Model(inputs=[inp],outputs=[out])
    model.compile(loss=cmae, metrics='mse')
    
    # TODO:set weights
    x  = np.arange(-2**(N_scale-1)+1, 2**(N_scale-1))
    y = x
    X,Y = np.meshgrid(x,y)
    polys=[]
    for kx in range(N_scale):
        for ky in range(N_scale):
            poly = np.multiply(np.power(x,kx), np.power(y,ky))
            polys.append(poly)
    

def train():
    N_features = 8
    N_x = int(np.math.ceil(np.sqrt(N_features)))
    base_dir = os.path.dirname(__file__)
    # base_dir = '/content/ImageDecomposition/scripts'
    model_fname = os.path.join(base_dir, f'sem_model_{N_features}.h5')
    img_fname = os.path.join(base_dir,'..','data','sem_images','SRAM_22nm.jpg')
    dg = DataGenerator(img_fname)

    model = create_model(N_features=N_features)

    base_model = create_base_model()
    if os.path.isfile(model_fname):
        model.load_weights(model_fname)

    model.fit(dg,epochs=100, verbose=1)
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


def analyse_model():
    base_dir =os.path.dirname(__file__)
    N_features = 8
    model_fname = os.path.join(base_dir, f'sem_model_{N_features}.h5')
    img_fname = os.path.join(base_dir,'..','data','sem_images','SRAM_22nm.jpg')
    dg = DataGenerator(img_fname)

    if not os.path.isfile(model_fname):
      print(f'model not found:{model_fname}')
      return

    N_x = int(np.math.ceil(np.sqrt(N_features)))
    
    model = create_model(N_features=N_features)
    model.load_weights(model_fname)
    # model = tf.keras.models.load_model(model_fname)

    # analyse the model
    features = model.get_layer('weight_sum').input
    regions = model.get_layer('regions').output
    debug_model = tf.keras.models.Model(inputs=[model.input], outputs=[features,regions, model.output])
    features,regions, out_img = debug_model.predict(dg.img4d)
    print(features.shape)
    print(regions.shape)
    plt.figure(1)
    for i in range(N_features):
      plt.subplot(N_x,N_x,i+1)
      plt.imshow(regions[0,:,:,i])

    plt.figure(2)
    for i in range(N_features):
      plt.subplot(N_x,N_x,i+1)
      plt.imshow(features[0,:,:,i])

    img = dg.img4d[0,:,:,0]
    gt_img = img
    
    # verify the model
    out_img4d = model.predict(dg.img4d)
    out_img = out_img4d[0,:,:,0]

    diff = out_img-gt_img
    mae = np.mean(np.abs(diff[15:-15,15:-15]))
    print(mae)
    # histogram
    diff_hist = np.histogram(np.abs(diff), bins=100)

    # base line
    base_model = create_base_model(5)
    
    base_out_img4d = base_model.predict(dg.img4d)
    base_out_img = base_out_img4d[0,:,:,0]

    base_diff = base_out_img-gt_img
    base_mae = np.mean(np.abs(base_diff[15:-15,15:-15]))
    print(base_mae)

    plt.figure(3)
    ax1=plt.subplot(231)
    plt.imshow(img)
    plt.subplot(232, sharex=ax1, sharey=ax1),plt.imshow(gt_img, vmin=0, vmax=1)
    plt.subplot(233, sharex=ax1, sharey=ax1),plt.imshow(out_img, vmin=0, vmax=1)
    plt.subplot(234, sharex=ax1, sharey=ax1),plt.imshow(diff, vmin=-0.2, vmax=0.2),plt.title(f'{mae:0.04f}')
    plt.subplot(235, sharex=ax1, sharey=ax1),plt.imshow(base_out_img, vmin=0, vmax=1)
    plt.subplot(236, sharex=ax1, sharey=ax1),plt.imshow(base_diff, vmin=-0.2, vmax=0.2),plt.title(f'{base_mae:0.04f}')

    plt.figure(4)
    plt.plot(diff_hist[1][:-1], diff_hist[0]),plt.grid(True)
    plt.show()

def analyze_loss():
    base_dir =os.path.dirname(__file__)
    N_features = 8
    model_fname = os.path.join(base_dir, f'sem_model_{N_features}.h5')
    img_fname = os.path.join(base_dir,'..','data','sem_images','SRAM_22nm.jpg')
    dg = DataGenerator(img_fname)

    if not os.path.isfile(model_fname):
      print(f'model not found:{model_fname}')
      return

    N_x = int(np.math.ceil(np.sqrt(N_features)))
    
    model = create_model(N_features=N_features)
    model.load_weights(model_fname)
    
    out_img4d = model.predict(dg.img4d)
    
    loss = cmae(out_img4d, dg.img4d)
    print(loss)
    pass

if __name__ == '__main__':
    # train()
    analyse_model()
    # analyze_loss()