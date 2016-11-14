from scipy.ndimage.morphology import morphological_gradient
import matplotlib.pyplot as plt
import numpy as np


def Contours(bin_image, contour_size=3):
    # Computes the contours
    grad = morphological_gradient(bin_image, size=(contour_size, contour_size))
    return grad


def ImageSegmentationSave(RGB, Segmentation, output):

    plt.imshow(RGB)
    ContourSegmentation = Contours(Segmentation)
    x_, y_ = np.where(ContourSegmentation > 0)
    plt.scatter(x=x_, y=y_, c='r', s=1)

    plt.savefig(output)
    plt.clf()


def ImageSegmentationShow(RGB, Segmentation):

    plt.imshow(RGB)
    ContourSegmentation = Contours(Segmentation)
    x_, y_ = np.where(ContourSegmentation > 0)
    plt.scatter(x=x_, y=y_, c='r', s=1)

    plt.show()
