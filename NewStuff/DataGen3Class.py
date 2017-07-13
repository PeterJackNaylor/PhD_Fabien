from DataGenRandomT import DataGenRandomT
import nibabel as ni
from Deprocessing.Morphology import generate_wsl
from skimage import measure
from scipy.ndimage.morphology import morphological_gradient

def Contours(bin_image, contour_size=3):
    # Computes the contours
    grad = morphological_gradient(bin_image, size=(contour_size, contour_size))
    return grad

class DataGen3(DataGenRandomT):

    def LoadGT(self, path):
        image = ni.load(path)
        img = image.get_data()
        img = measure.label(img)
        wsl = generate_wsl(img[:,:,0])
        img[ img > 0 ] = 1
        wsl[ wsl > 0 ] = 1
        img[:,:,0] = img[:,:,0] - wsl
        if len(img.shape) == 3:
            img = img[:, :, 0].transpose()
        else:
            img = img.transpose()
        cell_border = Contours(img, contour_size=3)
        img[cell_border > 0] = 2
        return img