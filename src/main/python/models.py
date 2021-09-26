import tensorflow as tf
import numpy as np
import os
import sys


# create Gaussian model
class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, half_kernel_size: tuple[int, int] = (5, 5), **kwargs):
        super().__init__(**kwargs)
        self.half_kernel_size = half_kernel_size

    def build(self, input_shape):
        super().build(input_shape)
        self.sigma = self.add_weight(name="sigma", shape=(2,), dtype=tf.float32, initializer=tf.initializers.constant(100.0), trainable=True,
                                     regularizer=None,
                                     constraint=tf.keras.constraints.non_neg())
        xx, yy = np.meshgrid(np.arange(-self.half_kernel_size[0], self.half_kernel_size[0]+1),
                             np.arange(-self.half_kernel_size[1], self.half_kernel_size[1]+1))
        xx = xx.reshape(xx.shape[0], xx.shape[1], 1, 1)
        yy = yy.reshape(yy.shape[0], yy.shape[1], 1, 1)
        self.xx = tf.constant(xx**2/2, dtype=tf.float32)
        self.yy = tf.constant(yy**2/2, dtype=tf.float32)

    def call(self, inputs):
        kernel = tf.exp(-(self.xx/(self.sigma[0]**2 + 1e-4)
                          ) - (self.yy/(self.sigma[1]**2 + 1e-4)))
        kernel_normalized = kernel / \
            tf.reduce_sum(kernel, keepdims=True, axis=[0, 1])
        return tf.nn.conv2d(inputs, kernel_normalized, strides=[1, 1, 1, 1], padding="SAME")

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "half_kernel_size": self.half_kernel_size
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def create_model_from_layer(layer):
    model = tf.keras.Sequential()
    model.add(layer)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.01))
    return model

def test_gaussian_layer():
    import cv2
    from matplotlib import pyplot as plt
    x = np.zeros((1, 11, 11, 1))
    x[0, 5, 5, 0] = 1
    x_tf = tf.constant(x)
    x_np = x[0, :, :, 0]
    gauss = cv2.getGaussianKernel(5, 1)
    gauss2d = gauss * gauss.T

    gt_y_np = cv2.GaussianBlur(x_np, (5, 5), 1)
    gt_y_np = gt_y_np.reshape(1, gt_y_np.shape[0], gt_y_np.shape[1], 1)
    gt_y_tf = tf.constant(gt_y_np)
    layer = GaussianLayer(half_kernel_size=(2, 2))
    model = create_model_from_layer(layer)
    y_tf = model(x_tf)
    model.layers[0].set_weights([2.0*np.ones((2,))])
    init_loss = model.evaluate(x_tf, gt_y_tf)
    print("Initial loss:", init_loss)
    model.fit(x_tf, gt_y_tf, epochs=300)
    final_loss = model.evaluate(x_tf, gt_y_tf)
    print("final loss:", final_loss)
    print(model.layers[0].get_weights())
    y_tf = model(x_tf)
    
    
    assert y_tf.shape == gt_y_tf.shape
    # assert np.allclose(y.numpy()[0, :, :, 0], gt_y_tf.numpy()[0, :, :, 0], atol=1e-4)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(x_np, cmap="gray")
    axs[0, 0].set_title("x")
    axs[0, 1].imshow(gt_y_np[0, :, :, 0], cmap="gray")
    axs[0, 1].set_title(f"gt_y: {gt_y_np.sum()}")
    axs[1, 0].imshow(y_tf.numpy()[0, :, :, 0], cmap="gray")
    axs[1, 0].set_title(f"y: {y_tf.numpy().sum()}")
    axs[1, 1].imshow(y_tf.numpy()[0, :, :, 0] - gt_y_tf.numpy()[0, :, :, 0], cmap="gray")
    axs[1, 1].set_title("diff")
    plt.show()
    
if __name__ == "__main__":
    test_gaussian_layer()