import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_poly(ksize, N_power=4):
    if N_power<=1:
        N_power = 1
    # N_features = N_power**2
    # kernel_size = [2**N_scale-1, 2**N_scale-1] 
    strides = (1, 1)
    
    kw = ksize
    kh = ksize

    x_range = np.arange(-int(kw//2), int(kw//2))/strides[1]
    y_range = np.arange(-int(kh//2), int(kh//2))/strides[0]
    xx,yy = np.meshgrid(x_range, y_range)

    # I = X*A
    # A = inv(XtX)*I
    # I (kh*kw,1)
    # X (kh*kw, Nx*Ny)
    # A (Nx*Ny,1)
    kernels_fw = np.zeros((kh*kw, N_power*N_power))
    kernel_norm = kw*kh
    for ky in range(N_power):
        for kx in range(N_power):
            k = ky*N_power + kx
            kernel = np.multiply(np.power(xx, kx),np.power(yy, ky))
            kernel = np.divide(kernel, kernel_norm)
            kernels_fw[:,k] = kernel.flatten()

    kernel_bk = np.matmul(np.linalg.inv(np.matmul(kernels_fw.T,kernels_fw)),kernels_fw.T)

    return kernels_fw, kernel_bk

    base_dir =os.path.dirname(__file__)
    img_fname = os.path.join(base_dir,'..','data','sem_images','SRAM_22nm.jpg')
    img_ori = cv2.imread(img_fname,0).astype(float)/255.0

    offset = [160,90]
    I = img_ori[offset[0]:offset[0]+kh,offset[0]:offset[0]+kw]
    I = I.reshape((-1,1))
    A = np.matmul(kernel_bk, I)
    recon_I =  np.matmul(kernels_fw, A)

    img = I.reshape((kh,kw))
    recon_img = recon_I.reshape((kh,kw))
    diff_img = recon_img- img
    ax1= plt.subplot(231)
    plt.imshow(img, vmin=0, vmax=1)
    plt.subplot(232, sharex=ax1, sharey=ax1),plt.imshow(recon_img, vmin=0, vmax=1)
    plt.subplot(233, sharex=ax1, sharey=ax1),plt.imshow(diff_img, vmin=-0.3, vmax=0.3)
    plt.subplot(224),plt.imshow(img_ori)
    plt.show()

def test_create_polys():
    N = 64
    img = np.zeros((N,N))
    img[:,:N//2]=  1
    kf,kb = create_poly(ksize=N, N_power=6)

    fimg = np.matmul(kb,img.reshape((-1,1)))
    print(fimg)
    rimg = np.matmul(kf,fimg).reshape((N,N))
    plt.imshow(rimg),plt.show()
    print(kf.shape)
    print(kb.shape)
    pass


if __name__ == '__main__':
    test_create_polys()
    