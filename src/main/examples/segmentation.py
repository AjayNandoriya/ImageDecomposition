import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_marker(img):
    bg_dist_th, fg_dist_th = 0.5, 0.8
    # Threshold the image to create a binary mask
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Perform distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

    # Threshold the distance transform to create markers
    fg_mask = dist_transform>fg_dist_th*dist_transform.max()

    # Perform morphological opening to remove small objects
    closing = cv2.bitwise_not(opening)
    dist_transform_neg = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
    bg_mask = dist_transform_neg > bg_dist_th*dist_transform_neg.max()
    prob_bg_mask = (dist_transform_neg >0) & ~(bg_mask)
    prob_fg_mask = (dist_transform >0) & ~fg_mask
    
    
    init_mask = bg_mask*0 + fg_mask*1 +  2*prob_bg_mask + 3*prob_fg_mask
    init_mask = init_mask.astype(np.uint8)
    return init_mask
    


def test_graphcut():
    # Load image
    img = cv2.imread(r'C:\dev\repos\ImageDecomposition\src\main\resources\sem_images\SRAM_22nm.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    img = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create binary masks for foreground and background regions
    fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    rect = (50, 50, 300, 300)
    cv2.rectangle(bg_mask, (50, 50), (350, 350), (255, 255, 255), thickness=-1)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # Define the region of interest (ROI) using a rectangle
    # The ROI should contain both foreground and background regions
    # cv2.grabCut(img, bg_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    init_mask = get_marker(img)
     
    # Create a binary mask for the probable foreground region
    prob_fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    prob_fg_mask[(fg_mask == 1) | (bg_mask == 1)] = 1

    prob_fg_mask = init_mask.copy()
    # Refine the segmentation using GraphCut algorithm
    cv2.grabCut(np.tile(img[:,:,np.newaxis],[1,1,3]), prob_fg_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)

    # Create a binary mask for the final segmentation
    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    final_mask[(prob_fg_mask == 1) | (prob_fg_mask == 3)] = 255

    # Apply the mask to the original image
    segmented_img = cv2.bitwise_and(img, img, mask=final_mask)

    colormap = np.arange(256, dtype=np.uint8)
    colormap[:4] = [0,3,1,2]
    prob_fg_mask_map = cv2.applyColorMap(prob_fg_mask, colormap)
    init_mask_map = cv2.applyColorMap(init_mask, colormap)
    # Display the result
    fig,axes = plt.subplots(1,3,sharex=True, sharey=True)
    axes[0].imshow(init_mask_map)
    axes[1].imshow(prob_fg_mask_map)
    axes[2].imshow(img)
    plt.show()

    pass

def test_watershed():
    # Load the SEM image
    img = cv2.imread(r'C:\dev\repos\ImageDecomposition\src\main\resources\sem_images\SRAM_22nm.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)


    # Threshold the image to create a binary mask
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Perform distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

    # Threshold the distance transform to create markers
    ret, markers = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, cv2.THRESH_BINARY)

    # Apply connected component analysis to label the markers
    markers = cv2.connectedComponents(markers.astype(np.uint8))[1]

    # Apply watershed algorithm
    markers = cv2.watershed(np.tile(blur[:,:,np.newaxis],[1,1,3]), markers)

    # Overlay the segmented regions onto the original image
    img[markers == -1] = [0, 0, 255]  # Mark the boundary regions in red

    # Display the result
    fig,axes = plt.subplots(1,3,sharex=True, sharey=True)
    axes[0].imshow(opening)
    axes[1].imshow(dist_transform)
    axes[2].imshow(img)
    plt.show()


def test_watershed_clustering():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import label

    # Load the SEM image
    img = cv2.imread(r'C:\dev\repos\ImageDecomposition\src\main\resources\sem_images\SRAM_22nm.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    image = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find the local maxima of the gradient
    edge_peak_coords = peak_local_max(-image, min_distance=50, labels=image)

    mask = np.zeros(image.shape, dtype=bool)
    mask[tuple(edge_peak_coords.T)] = True
    # Label the markers using connected component analysis
    markers, num_markers = label(mask)

    # Perform watershed hierarchical clustering
    labels = watershed(image, markers, watershed_line=True)

    # Display the result
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3))

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original image')

    ax2.imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    ax2.set_title('Segmented image')

    ax3.imshow(mask, cmap=plt.cm.gray)
    ax3.set_title('Mask')


    # for ax in (ax1, ax2):
    #     ax.axis('off')

    plt.tight_layout()
    plt.show()

def test_superpixel():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.segmentation import slic
    from skimage import data

    # Load the SEM image
    img = cv2.imread(r'C:\dev\repos\ImageDecomposition\src\main\resources\sem_images\SRAM_22nm.jpg')
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    image = cv2.GaussianBlur(gray, (5, 5), 0)
    image = np.tile(image[:,:,np.newaxis],[1,1,3])
    
    # image = data.astronaut()
    
    # Perform superpixel segmentation
    segments = slic(image, n_segments=400, compactness=50, channel_axis=-1)

    # Display the result
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))

    ax1.imshow(image)
    ax1.set_title('Original image')

    ax2.imshow(segments, cmap='nipy_spectral')
    ax2.set_title('Superpixel segmentation')

    for ax in (ax1, ax2):
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_graphcut()
    # test_watershed()
    # test_watershed_clustering()
    # test_superpixel()
    pass