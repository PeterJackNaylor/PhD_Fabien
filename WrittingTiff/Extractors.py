import numpy as np
from skimage.morphology import watershed, dilation, disk, reconstruction
from skimage.measure import regionprops, label
import pandas as pd
from scipy.misc import imread
from skimage.color import rgb2gray
import pdb
from scipy.misc import imsave

def bin_analyser(RGB_image, bin_image, list_feature, marge=None, pandas_table=False, do_label=True):
    bin_image_copy = bin_image.copy()
    if marge is not None and marge != 0:
        seed = np.zeros_like(bin_image_copy)
        seed[marge:-marge, marge:-marge] = 1
        mask = bin_image_copy.copy()
        mask[ mask > 0 ] = 1
        mask[marge:-marge, marge:-marge] = 1
        reconstructed = reconstruction(seed, mask, 'dilation')
        bin_image_copy[reconstructed == 0] = 0
    if do_label:
        bin_image_copy = label(bin_image_copy)

    if len(np.unique(bin_image_copy)) != 2:
        if len(np.unique(bin_image_copy)) == 1:
            if 0 in bin_image_copy:
                print "Return blank matrix. Change this shit"
                return np.array([[0, 0, 0, 0, 0]])
            else:
                print "Error, must give a bin image."
    
    GrowRegion_N = NeededGrownRegion(list_feature)
    img = {0: bin_image_copy}
    RegionProp = {0: regionprops(bin_image_copy)}
    for val in GrowRegion_N:
        if val != 0:
            img[val] = GrowRegion(bin_image_copy, val)
            RegionProp[val] = regionprops(img[val])

    n = len(RegionProp[0])
    p = 0
    for feat in list_feature:
        p += feat.size
    TABLE = np.zeros(shape=(n,p))
    for i in range(n):
        offset_ALL = 0
        for j, feat in enumerate(list_feature):
            tmp_regionprop = RegionProp[feat._return_n_extension()][i]
            off_tmp = feat.size      
            TABLE[i, (j + offset_ALL):(j + offset_ALL + off_tmp)] = feat._apply_region(tmp_regionprop ,RGB_image)

            offset_ALL += feat.size - 1

    if pandas_table:
        names = []
        for el in list_feature:
            if el.size != 1:
                for it in range(el.size):
                    names.append(el._return_name()[it])
            else:
                names.append(el._return_name())
        return pd.DataFrame(TABLE, columns=names)
    else:
        return TABLE


def bin_to_color(RGB_image, bin_image, feat_vec, marge=None, do_label=True):
    bin_image_copy = bin_image.copy()
    res = np.zeros(shape=(bin_image.shape[0], bin_image.shape[1], 3))
    if marge is not None:
        seed = np.zeros_like(bin_image_copy)
        seed[marge:-marge, marge:-marge] = 1
        mask = bin_image_copy.copy()
        mask[ mask > 0 ] = 1
        mask[marge:-marge, marge:-marge] = 1
        reconstructed = reconstruction(seed, mask, 'dilation')
        bin_image_copy[reconstructed == 0] = 0
    if do_label:
        bin_image_copy = label(bin_image_copy)

    if len(np.unique(bin_image_copy)) != 2:
        if len(np.unique(bin_image_copy)) == 1:
            if 0 in bin_image_copy:
                print "Return blank matrix."
                return bin_image_copy
            else:
                print "Error, must give a bin image."
    RegProp = regionprops(bin_image_copy)
    for i in range(len(RegProp)):
        mini_reg = RegProp[i]
        bin_image_copy[mini_reg.coords] = feat_vec[i]
    return bin_image_copy

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
        self.size = 0
        self.GetSize()
    def _return_size(self):
        return self.size
    def _return_name(self):
        return self.name
    def _return_n_extension(self):
        return self.n_extension
    def _apply_region(self, regionp, RGB):
        raise NotImplementedError
    def GetSize(self):
        raise NotImplementedError
class PixelSize(Feature):
    def _apply_region(self, regionp, RGB):
        bin = regionp.image.astype(np.uint8)
        return Pixel_size(bin)
    def GetSize(self):
        self.size = 1

class Save(Feature):
    """Not usable yet, need to figure out a way for the diff name"""
    def _apply_region(self, regionp, RGB):
        bin = regionp.image.astype(np.uint8)
        x_m, y_m, x_M, y_M = regionp.bbox
        img_crop = RGB[x_m:x_M, y_m:y_M]
        imsave("test_rgb.png" ,img_crop)
        imsave("test.png", bin)
        pdb.set_trace()
    def GetSize(self):
        self.size = 0

class MeanIntensity(Feature):
    def _apply_region(self, regionp, RGB):
        bin = regionp.image.astype(np.uint8)
        x_m, y_m, x_M, y_M = regionp.bbox
        img_crop = RGB[x_m:x_M, y_m:y_M]
        return Mean_intensity(img_crop, bin)
    def GetSize(self):
        self.size = 1

class Centroid(Feature):
    def _apply_region(self, regionp, RGB):
        return regionp.centroid
    def GetSize(self):
        self.size = 2

list_f = [PixelSize("Pixel_sum", 0), MeanIntensity("Intensity_mean_0", 0), 
          MeanIntensity("Intensity_mean_5", 5), Centroid(["Centroid_x", "Centroid_y"], 0)]

list_f_names = []
for el in list_f:
    if el.size == 1:
        list_f_names.append(el.name)
    else:
        for i in range(el.size):
            list_f_names.append(el.name[i])

if __name__ == '__main__':
    img = imread("/Users/naylorpeter/prob/CIT_1_40x_raw.png")
    img_bin = imread("/Users/naylorpeter/prob/bin_img2.png")
    list_f = [PixelSize("Pixel sum", 0), MeanIntensity("Intensity mean 0", 0), 
              MeanIntensity("Intensity mean 5", 5)]
    table_feat = bin_analyser(img, img_bin, list_f, pandas_table=True)
