import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import cv2

@tf.custom_gradient
def get_gaussian_circular_kernel_2d(r2, sigma, gain):
    """
    Returns a 2D Gaussian kernel with a circular shape.
    """
    sigma_sign = tf.sign(sigma)
    SIGMA_MIN = tf.constant(0.4)
    sigma = tf.abs(sigma)
    kernel_gain1 = tf.exp(-r2 / (tf.constant(2.0) * tf.square(sigma)))
    kernel = gain*kernel_gain1
    def grad(dy):
        kernel_shape = tf.shape(kernel)
        half_ksize = tf.cast(kernel_shape[0]/2, tf.int64)
        dgain1 = tf.reduce_sum(tf.multiply(dy, kernel_gain1))

        kernel_gain1_sigma_min = tf.exp(-r2 / (tf.constant(2.0) * tf.square(SIGMA_MIN)))
        dgain2 = tf.reduce_sum(tf.multiply(dy, kernel_gain1_sigma_min))
        dgain = (dgain1*sigma + dgain2*SIGMA_MIN)/(sigma + SIGMA_MIN)
        dgain = dgain1
        dgain = dy[half_ksize, half_ksize, 0, 0]*0.1
        dr2 = tf.zeros_like(r2)
        # dsigma = tf.zeros_like(sigma)
        dsigma = tf.reduce_sum(tf.multiply(dy, kernel* r2))
        dsigma1 = tf.math.divide_no_nan(dsigma, tf.math.pow(sigma, 3))

        sp_delta = tf.SparseTensor(
            dense_shape=tf.cast(kernel_shape, dtype=tf.int64),
            values=[1.0],
            indices =[[half_ksize, half_ksize, 0, 0]])

        delta = tf.sparse.to_dense(sp_delta)
        kernel_gain1_sigma_min =  kernel_gain1_sigma_min - delta
        dsigma2 = kernel_gain1_sigma_min*(gain)/ SIGMA_MIN
        dsigma2 = tf.reduce_sum(tf.multiply(dy, dsigma2))
        dsigma = (dsigma1*sigma + dsigma2*SIGMA_MIN)/(sigma + SIGMA_MIN)
        dsigma = sigma_sign*dsigma
        return dr2, dsigma, dgain
    return kernel, grad

def test_gaussian():
    half_ksize = 2
    sigma = 0.4
    gain = 0.1
    x,y = tf.meshgrid(tf.range(-half_ksize, half_ksize+1), tf.range(-half_ksize, half_ksize+1))
    r2 = tf.square(x) + tf.square(y)
    r2 = tf.cast(r2, tf.float32)
    r2 = tf.expand_dims(tf.expand_dims(r2, axis=-1), axis=-1)

    kernel = get_gaussian_circular_kernel_2d(r2, sigma, gain)
    print(kernel.numpy()[:,:,0,0])

@tf.function
def gaussian_circular(inputs, half_ksize, sigma, gain):
    x,y = tf.meshgrid(tf.range(-half_ksize, half_ksize+1), tf.range(-half_ksize, half_ksize+1))
    r2 = tf.square(x) + tf.square(y)
    r2 = tf.cast(r2, tf.float32)
    r2 = tf.expand_dims(tf.expand_dims(r2, axis=-1), axis=-1)

    kernel = get_gaussian_circular_kernel_2d(r2, sigma, gain)
    
    inp_padded = tf.pad(inputs, [[0,0],[half_ksize, half_ksize],[half_ksize, half_ksize],[0,0]], mode='SYMMETRIC')
    output = tf.nn.conv2d(inp_padded, kernel, strides=[1,1,1,1], padding='VALID')
    return output

class GaussianCircular(tf.keras.layers.Layer):
    """2D Circular Gaussian filter """
    def __init__(self, half_ksize, padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self.half_ksize = half_ksize
        self.padding = padding
        
    def build(self, input_shape):
        sigma_mean = self.half_ksize*0.3 +0.16
        x,y = tf.meshgrid(tf.range(-self.half_ksize, self.half_ksize+1), tf.range(-self.half_ksize, self.half_ksize+1))
        self.r2 = tf.square(x) + tf.square(y)
        self.r2 = tf.cast(self.r2, tf.float32)
        self.r2 = tf.expand_dims(tf.expand_dims(self.r2, axis=-1), axis=-1)
        self.sigma = self.add_weight(name='sigma', shape=(), initializer=tf.keras.initializers.TruncatedNormal(sigma_mean, 0.05), trainable=True)
        self.gain = self.add_weight(name='gain', shape=(), initializer=tf.keras.initializers.TruncatedNormal(1.0, 0.05), trainable=True)

    def call(self, inputs):
        kernel = get_gaussian_circular_kernel_2d(self.r2, self.sigma, self.gain)
        if self.padding == 'SAME':
            output = tf.nn.conv2d(inputs, kernel, strides=[1,1,1,1], padding='SAME')
        elif self.padding == 'VALID':
            output = tf.nn.conv2d(inputs, kernel, strides=[1,1,1,1], padding='VALID')
        elif self.padding == 'SYMMETRIC':
            inp_padded = tf.pad(inputs, [[0,0],[self.half_ksize,self.half_ksize],[self.half_ksize,self.half_ksize],[0,0]], mode='SYMMETRIC')
            output = tf.nn.conv2d(inp_padded, kernel, strides=[1,1,1,1], padding='VALID')
        else:
            raise ValueError('Invalid padding mode')
        return output

    def compute_output_shape(self, input_shape):
        if self.padding == 'SAME' or self.padding == 'SYMMETRIC':
            return tf.TensorShape(input_shape)
        elif self.padding == 'VALID':
            return tf.TensorShape([input_shape[0]] + [input_shape[1:3]-2*self.half_ksize] + [input_shape[3]])
        else:
            raise ValueError('Invalid padding mode')

    def get_config(self):
        config = super().get_config()
        config.update({
            'half_ksize': self.half_ksize,
            'padding': self.padding
        })
        return config

class Conv2D(tf.keras.layers.Layer):
    """2D filter """
    def __init__(self, half_ksize, padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self.half_ksize = half_ksize
        self.padding = padding
        
    def build(self, input_shape):
        ksize = (2*self.half_ksize+1, 2*self.half_ksize+1, 1, 1)
        self.kernel = self.add_weight(name='sigma', shape=ksize, initializer=tf.keras.initializers.TruncatedNormal(1, 0.05), trainable=True)
        
    def call(self, inputs):
        if self.padding == 'SAME':
            output = tf.nn.conv2d(inputs, self.kernel, strides=[1,1,1,1], padding='SAME')
        elif self.padding == 'VALID':
            output = tf.nn.conv2d(inputs, self.kernel, strides=[1,1,1,1], padding='VALID')
        elif self.padding == 'SYMMETRIC':
            inp_padded = tf.pad(inputs, [[0,0],[self.half_ksize,self.half_ksize],[self.half_ksize,self.half_ksize],[0,0]], mode='SYMMETRIC')
            output = tf.nn.conv2d(inp_padded, self.kernel, strides=[1,1,1,1], padding='VALID')
        else:
            raise ValueError('Invalid padding mode')
        return output

    def compute_output_shape(self, input_shape):
        if self.padding == 'SAME' or self.padding == 'SYMMETRIC':
            return tf.TensorShape(input_shape)
        elif self.padding == 'VALID':
            return tf.TensorShape([input_shape[0]] + [input_shape[1:3]-2*self.half_ksize] + [input_shape[3]])
        else:
            raise ValueError('Invalid padding mode')

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel': self.kernel
        })
        return config

def cmae(y_true, y_pred, eps=0.01):
    diff = tf.abs(y_true-y_pred)
    diff = tf.maximum(diff, eps)
    return tf.reduce_mean(diff)


def test_gaussian_train():

    # inputs
    half_ksize = 2

    # data
    # GT
    sigma = 2
    ksize = (2*half_ksize + 1, 2*half_ksize + 1)
    img = np.zeros((128,128), dtype=np.float32)
    img[32:96,32:96] = 1.0
    gt_img = cv2.GaussianBlur(img, ksize, sigma).astype(np.float32)
    img_4d= img.reshape(1,128,128,1)
    gt_img_4d = gt_img.reshape(1,128,128,1)

    inputs = tf.constant(img_4d)
    y_true = tf.constant(gt_img_4d)

    # initialization
    x,y = tf.meshgrid(tf.range(-half_ksize, half_ksize+1), tf.range(-half_ksize, half_ksize+1))
    r2 = tf.square(x) + tf.square(y)
    r2 = tf.cast(r2, tf.float32)
    r2 = tf.expand_dims(tf.expand_dims(r2, axis=-1), axis=-1)

    
    inp_padded = tf.pad(inputs, [[0,0],[half_ksize, half_ksize],[half_ksize, half_ksize],[0,0]], mode='SYMMETRIC')

    # init variables
    sigma = tf.Variable(1.0, trainable=True)
    gain = tf.Variable(0.24, trainable=True)
    optimizer = tf.keras.optimizers.Adam(lr=0.1)
    epochs = 1000        
    trainable_variables = [sigma, gain]
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        with tf.GradientTape() as tape:
            kernel = get_gaussian_circular_kernel_2d(r2, sigma, gain)
            y_pred = tf.nn.conv2d(inp_padded, kernel, strides=[1,1,1,1], padding='VALID')
            loss_value = cmae(y_true, y_pred)
        monitor_nodes = trainable_variables + [kernel, y_pred]
        grads = tape.gradient(loss_value, monitor_nodes)
        trainable_grads = grads[:len(trainable_variables)]
        extra_grads = grads[len(trainable_variables):]
        optimizer.apply_gradients(zip(trainable_grads, trainable_variables))
        print("Training loss (for one batch) at epoch %d: %.4f" % (epoch, float(loss_value)))
        print(f' grad: {trainable_grads}')
        print(f' weigths: {trainable_variables}')
        # print(f' kernel grad: {extra_grads[0].numpy()[:,:,0,0]}')
        # print(f' kernel: {kernel.numpy()[:,:,0,0]}')
        # print(f' out grad: {extra_grads[1].numpy()[0,:, :,0]}')
        
        plt.subplot(321),plt.imshow(y_pred.numpy()[0, :,:,0])
        plt.subplot(322),plt.imshow(y_pred.numpy()[0, :,:,0] - y_true.numpy()[0, :,:,0]), plt.colorbar()
        plt.subplot(323),plt.imshow(extra_grads[1].numpy()[0, :,:,0]), plt.colorbar()
        plt.subplot(324),plt.imshow(kernel.numpy()[:,:,0,0])
        plt.subplot(326),plt.imshow(extra_grads[0].numpy()[:,:,0,0]), plt.colorbar()
        # plt.show()
        pass
        

def train(model, x, y, epochs, batch_size, optimizer, loss_fn):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(zip([x],[y])):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, y_pred)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if epoch % 1 == 0:
            print("Training loss (for one batch) at epoch %d: %.4f" % (epoch, float(loss_value)))
            print(f' grad: {grads}')
            print(f' weigths: {model.trainable_weights}')
        pass 


def test_gaussian_circular_kernel_2d():
    """
    Test the Gaussian kernel with a circular shape.
    """
    tf.keras.backend.clear_session()
    sigma = 2
    gain = 1.0
    half_ksize = 3
    ksize = (2*half_ksize + 1, 2*half_ksize + 1)

    img = np.zeros((128,128), dtype=np.float32)
    img[32:96,32:96] = 1.0
    gt_img = cv2.GaussianBlur(img, ksize, sigma).astype(np.float32)


    model = tf.keras.models.Sequential([GaussianCircular(half_ksize=half_ksize, padding='SYMMETRIC')])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), loss=cmae)

    img_4d= img.reshape(1,128,128,1)
    gt_img_4d = gt_img.reshape(1,128,128,1)

    img_out_4d = model.predict(img_4d)
    # model.fit(img_4d, gt_img_4d, epochs=500, verbose=1)

    train(model, img_4d, gt_img_4d, epochs=500, batch_size=1, optimizer=tf.keras.optimizers.Adam(lr=0.1), loss_fn=cmae)

    img_out_4d = model.predict(img_4d)
    img_out = img_out_4d.reshape(128,128)
    img_diff = img_out-gt_img
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set_title('Input')
    axs[0,1].imshow(gt_img, cmap='gray')
    axs[0,1].set_title('Ground Truth')
    axs[1,0].imshow(img_out, cmap='gray')
    axs[1,0].set_title('Output')
    axs[1,1].imshow(img_diff, cmap='gray')
    axs[1,1].set_title(f'Difference {np.abs(img_diff).max()}')
    plt.show()

if __name__ == '__main__':
    # test_gaussian_circular_kernel_2d()
    test_gaussian_train()
    # test_gaussian()