import unittest
import image_kernels
import numpy as np

class TestImageKernels(unittest.TestCase):
    def test_get_avg_angular_kernel_0(self):
        """Test the get_avg_angular_kernel() function with 0 degree."""

        kernel_half_width = 5
        # Get the average angular kernel.
        avg_angular_kernel = image_kernels.get_avg_angular_kernel(kernel_half_width=kernel_half_width, angle=0, tight=True)

        gt_avg_angular_kernel = np.ones((1, 2*kernel_half_width+1))/(2*kernel_half_width+1)

        self.assertAlmostEqual(gt_avg_angular_kernel, avg_angular_kernel, delta=0.0001)
        
    def test_get_avg_angular_kernel_45(self):
        """Test the get_avg_angular_kernel() function with 45 degree."""
        kernel_half_width = 5
        # Get the average angular kernel.
        avg_angular_kernel = image_kernels.get_avg_angular_kernel(kernel_half_width=kernel_half_width, angle=45, tight=True)

        # gt_avg_angular_kernel = np.ones((1, 2*kernel_half_width+1))/(2*kernel_half_width+1)

        # diff = np.abs(gt_avg_angular_kernel - avg_angular_kernel).max()
        # self.assertAlmostEqual(gt_avg_angular_kernel, avg_angular_kernel, delta=0.0001)

    def test_get_avg_angular_kernel_90(self):
        """Test the get_avg_angular_kernel() function with 90 degree."""

        kernel_half_width = 5
        # Get the average angular kernel.
        avg_angular_kernel = image_kernels.get_avg_angular_kernel(kernel_half_width=kernel_half_width, angle=90, tight=True)

        gt_avg_angular_kernel = np.ones((2*kernel_half_width+1, 1))/(2*kernel_half_width+1)

        self.assertAlmostEqual(gt_avg_angular_kernel, avg_angular_kernel, delta=0.0001)
    
if __name__ == '__main__':
    unittest.main()