import sys
import glob
import os
from gi.repository import Vips
from UsefulFunctions.RandomUtils import CheckOrCreate
import numpy as np

def gather_files(folder):
    """ Find all tif files in folder"""
    name = os.path.join(folder, "*")
    return glob.glob(name)


def find_closest(list_image, x0, y0):
    """ Return coordinates of images closest to (x_0, y_0)"""
    def f(element):
        el = element.split('/')[-1].split('.')[0]
        x, y = el.split('_')
        return (int(x) - x0)**2 + (int(y) - y0)**2
    distance = map(f, list_image)
    return list_image[np.argmin(distance)]


def find_n_closest(list_image, x0, y0, n):
    """ Return n images closest to (x_0, y_0)"""
    def f(element):
        el = element.split('/')[-1].split('.')[0]
        x, y = el.split('_')
        return (int(x) - x0)**2 + (int(y) - y0)**2
    distanceX = map(f, list_image)
    rank_index = np.array(sorted(range(len(distanceX)), key=lambda k: distanceX[k]))[0:n]
    return list(np.array(list_image)[rank_index])


def in_square(x, y, X_interval, Y_interval):
    if x < X_interval[1] and x > X_interval[0]:
        if y < Y_interval[1] and y > Y_interval[0]:
            return True
    return False


def gather_whole_square(to_do, not_done):
    ''' given a list of images, return all images in the square'''
    def f(element):
        el = element.split('/')[-1].split('.')[0]
        x, y = el.split('_')
        return (x, y)
    coord = map(f, to_do)
    sort_by_first = sorted(coord, key=lambda tup: tup[0])
    sort_by_second = sorted(coord, key=lambda tup: tup[1])
    x_min = sort_by_first[0][0]
    x_max = sort_by_first[-1][0]
    y_min = sort_by_second[0][1]
    y_max = sort_by_second[-1][1]
    def g(element):
        el = element.split('/')[-1].split('.')[0]
        x, y = el.split('_')
        if in_square(x, y, [x_min, x_max], [y_min, y_max]):
            return el
    result = map(g, not_done)
    return result


def update(list_A, list_B):
    ''' removes all element A from B'''
    for el in list_A:
    	list_B.remove(el)
    return list_A, list_B


def construct_inter(list_image, save_fold):
    def f(element):
        el = element.split('/')[-1].split('.')[0]
        x, y = el.split('_')
        return (x, y)
    coord = map(f, list_image)
    sort_by_first = sorted(coord, key=lambda tup: tup[0])
    sort_by_second = sorted(coord, key=lambda tup: tup[1])
    x_min = sort_by_first[0][0]
    x_max = sort_by_first[-1][0]
    y_min = sort_by_second[0][1]
    y_max = sort_by_second[-1][1]

    size_x = x_max - x_min
    size_y = y_max - y_min 
    img = Vips.Image.black(size_x, size_y)
    for el in list_image:
        el = element.split('/')[-1].split('.')[0]
        x, y = el.split('_')
        tile = Vips.Image.new_from_file(el, 
                        access = Vips.Access.SEQUENTIAL_UNBUFFERED)
        img = img.insert(tile, int(x) - x_min, int(y) - y_min)
    outfile = "{}_{}_{}_{}.tif".format(x_min, y_min, size_x, size_y)
    outfile = os.path.join(save_fold, outfile)
    img.tiffsave(outfile, tile=True, pyramid=True, bigtiff = True) # change this line


def construct_exter(list_image, name):
    def f(element):
        el = element.split('/')[-1].split('.')[0]
        x, y, size_x, size_y = el.split('_')
        return (x, y, size_x, size_y)
    coord = map(f, list_image)
 
    sort_by_first = sorted(coord, key=lambda tup: tup[0])
    sort_by_furthest_x = sorted(coord, key=lambda tup: tup[0] + tup[2])
    
    sort_by_second = sorted(coord, key=lambda tup: tup[1])
    sort_by_furthest_y = sorted(coord, key=lambda tup: tup[1] + tup[3])

    x_min = sort_by_first[0][0]
    x_max = sort_by_furthest_x[-1][0] + sort_by_furthest_x[-1][2]
    y_min = sort_by_second[0][1]
    y_max = sort_by_furthest_y[-1][1] + sort_by_furthest_y[-1][3]

    size_x = x_max - x_min 
    size_y = y_max - y_min 

    img = Vips.Image.black(size_x, size_y)
    for el in list_image:
        el = element.split('/')[-1].split('.')[0]
        x, y, size_x, size_y = el.split('_')
        tile = Vips.Image.new_from_file(el, 
                        access = Vips.Access.SEQUENTIAL_UNBUFFERED)
        img = img.insert(tile, int(x) - x_min, int(y) - y_min)
    img.tiffsave(name, tile=True, pyramid=True, bigtiff = True) 


def __main__():
    slide_name = sys.argv[1]
    folder_img = sys.argv[2]
    temp_folder = './temp_folder'
    CheckOrCreate(temp_folder)
    n = sys.argv[3]
    image_list = gather_files(folder_img)

    first = True
    while len(image_list) != 0:
        name_tile = find_closest(image_list, 0, 0)
        x, y = name_tile.split('/')[-1].split('.')[0].split('_')
        to_analyse = find_n_closest(image_list, int(x), int(y), n)
        to_process = gather_whole_square(to_analyse, image_list)
        construct_inter(to_process, temp_folder)
        to_process, image_list = update(to_process, image_list)
        print len(image_list)

    image_list = gather_files(temp_folder)
    construct_exter(image_list, slide_name)


if __name__ == "__main__":
	__main__()
