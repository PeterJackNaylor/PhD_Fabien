import numpy as np
from skimage.morphology import watershed, dilation, disk
from skimage.measure import regionprops

def bin_analyser(bin_image, list_feature):
    


    bin_image = bin_image.astype(np.uint8)
    if len(np.unique(bin_image)) != 2:
        if len(np.unique(bin_image)) == 1:
            if 0 in bin_image:
                return bin_image
            else:
                print "Error, muszt give a bin image"
    
    GrowRegion_N = NeededGrownRegion(list_feature)
    img = {0: bin_image}
    RegionProp = {0: regionprops(bin_image)}
    for val in GrowRegion_N:
        if val != 0:
            img[val] = GrowRegion(bin_image, val)
            RegionProp[val] = regionprops(img[val])

    n = len(RegionProp[0])
    p = len(list_feature)

    TABLE = np.zeros(shape=(n,p))
    for val in GrowRegion_N:
        
    regionprops(GrowRegion(test,3))





def NeededGrownRegion(list_feature):
    res = []
    for feat in list_feature:
        if feat._return_n_extension() not in res:
            res += [feat._return_n_extension()]
    return res



def GrowRegion(bin_image, n_pix):
    op = disk(n_pix)
    dilated_mask = dilation(bin_image, selem=op)
    return  watershed(dilated_mask, bin_image, mask = dilated_mask)


def Pixel_size(img):
    return np.sum(img)

class Feature(object):
    """
    Generic python object for feature extraction from 
    a binary image.
    You can also override the name attribute.
    """
    def __init__(self, name, n_extension):
        self.name = name
        self.n_extension = n_extension

    def _return_name(self):
        return self.name
    def _return_n_extension(self):
        return self.n_extension
    def _apply_region(self, img):
        raise NotImplementedError

class PixelSize(Feature):
    def _apply_region(self, img):
        return Pixel_size(img)





list_f = [Pixel_size]