from UsefulFunctions.RandomUtils import CheckOrCreate
from optparse import OptionParser
import os
import numpy as np
import sys
sys.path.append("/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff")
from tifffile import imread, imsave
import pdb

from skimage.morphology import reconstruction, dilation, disk
import cv
import glob
def find_closest_cc(x, y, b):
    copy = np.zeros_like(b)
    copy[x, y] = 1
    dilat = 1
    found = False
    while not found:
        dilatedd = dilation(copy, disk(dilat))
        inter = AndImg(b, dilatedd)
        if np.sum(inter) != 0:
            found = True
        else:
            dilat += 1
    xx, yy = np.where(inter == 1)
    return xx[0], yy[0]

def AndImg(a, b):
    ## bin images
    if np.max(a) > 1 or np.max(b) > 1:
        raise NotImplementedError
    x, y = np.where(b == 1)
    tmp = a.copy()
    tmp[x, y] += 1
    tmp[tmp != 2] = 0
    tmp[tmp > 0] = 1
    return tmp
def find_bin(path):
        path = path.replace("_feat_0_color.npy", ".tiff")
        bin_name = os.path.join("/share/data40T_v2/Peter/PatientFolder/Job_579673/bin", path)
        return imread(bin_name)
def find_table(path):
        path = path.replace("_feat_0_color.npy", ".npy")
        table_name = os.path.join("/share/data40T_v2/Peter/PatientFolder/Job_579673/table", path)
        return np.load(table_name)
def change_color(table, bin, color_vec, indice, res):
    x_cent, y_cent = table[indice, 3:5]
    X, Y = int(x_cent), int(y_cent)
    only_cell = np.zeros_like(bin)
    if bin[X, Y] != 1:
        X, Y = find_closest_cc(X, Y, bin)


    only_cell[X, Y] = 1
    only_cell = reconstruction(only_cell, bin)
    x, y = np.where(only_cell == 1)
    res[x, y] = color_vec

from math import ceil
from UsefulFunctions.UsefulImageConstruction import sliding_window

def BlueRedGrad(val, min_val, max_val):
    alpha = float(val - min_val) / float(max_val - min_val)
    RGB = np.array([int(alpha * 255), 0, int((1 - alpha) * 255)])
    return RGB

def GradPourc(val):
    r, g, b = val
    alpha = float(r) / 255
    return alpha

def average(img):
    stepSize = 180
    windowSize = (180, 180)
    dim_x, dim_y, z = img.shape 
    if dim_x % windowSize[0] == 0:
        res_x = dim_x / stepSize
    else:
        res_x = dim_x / stepSize + 1
    if dim_y % windowSize[1] == 0:
        res_y = dim_y / stepSize
    else:
        res_y = dim_y / stepSize + 1
    res = np.zeros(shape=(res_x, res_y, 3))
    for x, y, width, height, window in sliding_window(img, stepSize, windowSize):

        X = ceil(float(x) / stepSize)
        Y = ceil(float(y) / stepSize)
        flattened = window.reshape((window.shape[0] * window.shape[1],3))
        Xi = flattened.sum(axis=1)
        if Xi.sum() == 0:
	     res_array = np.array([0, 0, 0])
	else:
             flattened = flattened[Xi != 0, :]
             alpha = np.mean(map(GradPourc, flattened))
             res_array = np.array([int(alpha * 255), 0, int((1 - alpha) * 255)])
        res[int(X), int(Y)] = res_array
    return res

from scipy.misc import imsave as imsave2

if __name__ == "__main__":
    name = glob.glob("./*_feat_0_color.npy")[0]
    table = find_table(name)
    bin = find_bin(name)
    for col_path in glob.glob("./*_feat_*_color.npy"):
        result = np.zeros(shape=(bin.shape[0], bin.shape[1], 3))
        feat_num = col_path[-11]
	def f((indice, val)):
	     change_color(table, bin, val, indice, result)
        map(f, enumerate(np.load(col_path)))
        CheckOrCreate("feat_{}".format(feat_num))
	imsave("feat_{}/".format(feat_num) + name.replace('_feat_0_color.npy', '.tiff'), result)
	imsave2("feat_{}/".format(feat_num) + name.replace('_feat_0_color.npy', '_mini.png'), average(result))
