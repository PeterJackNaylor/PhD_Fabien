import sys
import glob
import os


def gather_files(folder):
    """ Find all tif files in folder"""
    name = os.path.join(folder, "*")
    return glob.glob(name)

def find_closest(list_image, x0, y0):
    """ Return coordinates of images closest to (x_0, y_0)"""
    def f(element):
        el = element.split('/')[-1].split('.')[0]
        x, y = el.split('_')
        return ( (x- x0)**2 + (y-y0)**2)**0.5
    distance = map(f, list_image)
    return list_image[np.argmin(dist_image)]





