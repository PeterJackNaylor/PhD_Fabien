from UsefulFunctions.CuttingPatches import ROI
from optparse import OptionParser
import os
from tifffile import imsave, imread
from Extractors import bin_analyser, PixelSize, MeanIntensity, Centroid
import numpy as np
from skimage.morphology import dilation, erosion, disk

list_f = [PixelSize("Pixel_sum", 0), MeanIntensity("Intensity_mean_0", 0), 
          MeanIntensity("Intensity_mean_5", 5), Centroid(["Centroid_x", "Centroid_y"], 0)]

list_f_names = []
for el in list_f:
    if el.size == 1:
        list_f_names.append(el.name)
    else:
        for i in range(el.size):
            list_f_names.append(el.name[i])

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--bin", dest="bin",type="string",
                      help="bin name")
    parser.add_option("--rgb", dest="rgb",type="string",
                      help="rgb name")
    parser.add_option("--marge", dest="marge", type="int",
              help="how much to reduce the image size")
    (options, args) = parser.parse_args()

    rgb = imread(options.rgb)
    bin = imread(options.bin)

    table = bin_analyser(rgb, bin, list_f, options.marge, pandas_table=True)

    bin[bin > 0] = 1
    contours = dilation(bin, disk(2)) - erosion(bin, disk(2))
    x, y = np.where(contours == 1)
    image = rgb.copy()
    image[x, y] = np.array([0, 0, 0])
    imsave(options.rgb.replace('rgb', 'segmented'), image, resolution=[1.0,1.0])
    table.to_csv(options.bin.replace('rgb', 'table').repalce('tiff', 'csv'))