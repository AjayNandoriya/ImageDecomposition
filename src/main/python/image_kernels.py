import numpy as np
import cv2

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
