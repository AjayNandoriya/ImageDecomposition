import tensorflow as tf
import numpy as np


def test_tf2np():
    a_tf = tf.ones((10,))
    a_dl = tf.experimental.dlpack.to_dlpack(a_tf)
    a_np = np.from_dlpack(a_dl)
    pass


def test_np2tf():
    a_np = np.ones((10,))
    a_dl = a_np.to_dlpack()
    a_tf = tf.experimental.dlpack.from_dlpack(a_dl)
    pass


def test_tf_pipeline():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, (7, 7), (1, 1), padding='SAME', activation='relu')])

    model.compile(optimizer='adam', loss='mse')

    imgs_in = np.zeros((100, 128, 128, 1), np.float32)

    batch_size = 8
    dg = tf.data.Dataset.from_tensor_slices(imgs_in).batch(batch_size)

    out = np.empty(imgs_in.shape, np.float32)
    for i, batch in enumerate(dg):
        print(i, batch.shape)
        b0 = i * batch_size
        b1 = (i + 1) * batch_size
        out[b0:b1, :, :, :] = model.predict(batch)
        pass
    print(out.max())
    pass


if __name__ == '__main__':
    # test_tf2np()
    # test_np2tf()
    test_tf_pipeline()
    pass
