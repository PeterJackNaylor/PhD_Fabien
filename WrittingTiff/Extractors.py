import numpy as np
from skimage.morphology import watershed, dilation, disk
from skimage.measure import regionprops, label
import pandas as pd
from scipy.misc import imread
from skimage.color import rgb2gray
import pdb
from scipy.misc import imsave

def bin_analyser(RGB_image, bin_image, list_feature, pandas_table=False, do_label=True):
    if do_label:
        bin_image = label(bin_image)
    else:
        bin_image = bin_image

    if len(np.unique(bin_image)) != 2:
        if len(np.unique(bin_image)) == 1:
            if 0 in bin_image:
                print "Return blank matrix."
                return bin_image
            else:
                print "Error, must give a bin image."
    
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
    for i in range(n):
        for j, feat in enumerate(list_feature):
            tmp_regionprop = RegionProp[feat._return_n_extension()][i]
            bin_crop = tmp_regionprop.image.astype(np.uint8)
            x_m, y_m, x_M, y_M = tmp_regionprop.bbox
            img_crop = RGB_image[x_m:x_M, y_m:y_M]
            TABLE[i,j] = feat._apply_region(img_crop, bin_crop)
            
    if pandas_table:
        feature_name = [el._return_name() for el in list_feature]
        return pd.DataFrame(TABLE, columns=feature_name)
    else:
        return TABLE

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
    img[img > 0] = 1
    return np.sum(img)

def Mean_intensity(RGB, BIN):
    gray_rgb = rgb2gray(RGB)
    BIN = BIN > 0
    return np.mean(gray_rgb[BIN])

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
    def _apply_region(self, rgb_img, img_bin):
        raise NotImplementedError

class PixelSize(Feature):
    def _apply_region(self, rgb_img, img):
        return Pixel_size(img)


class Save(Feature):
    """Not usable yet, need to figure out a way for the diff name"""
    def _apply_region(self, rgb_img, img):
        imsave("test.png" ,img)
        imsave("test_rgb.png", rgb_img)
        pdb.set_trace()

class MeanIntensity(Feature):
    def _apply_region(self, rgb_img, img):
        return Mean_intensity(rgb_img, img)


if __name__ == '__main__':
    img = imread("/Users/naylorpeter/prob/CIT_1_40x_raw.png")
    img_bin = imread("/Users/naylorpeter/prob/bin_img2.png")
    list_f = [PixelSize("Pixel sum", 0), MeanIntensity("Intensity mean 0", 0), 
              MeanIntensity("Intensity mean 5", 5)]
    table_feat = bin_analyser(img, img_bin, list_f, pandas_table=True)
