import os
import sys

if __package__ is None or __package__ == '':
    PARENT_PATH = os.path.join(os.path.dirname(__file__),'..')
    sys.path.append(PARENT_PATH)

from imdecomposer import poly

def test_create_poly():
    N_scale = 4
    N_power = 4
    # gt_kernel_fw, gt_kernel_bk
    kernel_fw, kernel_bk = poly.create_poly(N_scale=N_scale, N_power=N_power)
    print(kernel_fw, kernel_bk)

if __name__ == '__main__':
    test_create_poly()

