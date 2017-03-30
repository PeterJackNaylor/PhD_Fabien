from UsefulOpenSlide import GetImage
from CuttingPatches import ROI, ROI_binary_mask, GetContours
import numpy as np
from optparse import OptionParser
import os
import pdb
from scipy.misc import imsave

def GetLabel(name, para):
    if "Normal" in name:
        return 0
    else:
        if "/" not in name:
            mask_name = os.path.join('/share/data40T_v2/CAMELYON16_data/Tumor_Mask/', name).replace(".tif", "_Mask.tif")
        else:
            mask_name = name.replace("/Tumor/", "/Tumor_Mask/").replace(".tif", "_Mask.tif")
        
        img = GetImage(mask_name, para)
        val, counts = np.unique(img, return_counts=True)
        if len(val) == 1 and val[0] == 255:
            return 1
        elif len(val) == 1 and val[0] == 0:
            return 0
        else:
            if val[np.argmax(counts)] == 255:
                return 1
            else:
                return 0
            

def CreateFileParam(name, list, file):
    """
    Creates physically a text file named name where each line as an id 
    and each line as parameters
    """
    f = open(name, "wb")
    for line, para in enumerate(list):

        Label = GetLabel(file, para)
        para = tuple(para)
        para += (file,)
        para += (Label,)
        pre = "{} {} {} {} {} {} {}\n".format(*para)

        f.write(pre)
    f.close()

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-f", "--file", dest="file",
                      help="Input file (raw data)")

    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="Where to store the patches")

    parser.add_option("-t", "--type", dest="method",
                      help="Expecting Tumor or Normal")
    parser.add_option("--ticket_val", dest="ticket_val", default=80, type="int",
                      help="Pixel value to cut ticket.")

    (options, args) = parser.parse_args()

    if "Tumor" in options.method:
        method = "SP_ROI_tumor"
    else:
        method = "SP_ROI_normal"

    basename = options.file
    name = basename
    name = os.path.join('/share/data40T_v2/CAMELYON16_data/{}', name)
    name = name.format(options.method)

    list_roi = ROI(name, ref_level=0, disk_size=4, thresh=230, black_spots=None,
        number_of_pixels_max=50176, verbose=False, marge=0, method=method,
        contour_size=3, N_squares=(200, 300, 50), seed=None, cut_whitescore=0.8,
        ticket_val=options.ticket_val)

    FILENAME = name.replace(".tif", ".txt")
    FILEPATH = os.path.join(options.output_folder, basename).replace('.tif', '.txt')
    CreateFileParam(FILEPATH, list_roi, basename)


    basename_png = basename.replace('.tif','') + "{}_{}_{}_{}_{}.png"
    FILEPATH = os.path.join(options.output_folder, basename_png)
    for para in list_roi:
        img = GetImage(name, para)
        imsave(FILEPATH.format(*para), img)
'''
    import openslide as op
    from scipy.misc import imsave
    slide = op.open_slide(name)
    lowest_res = len(slide.level_dimensions) - 2
    s = np.array(slide.read_region((0, 0), lowest_res,
                                       slide.level_dimensions[lowest_res]))[:, :, 0:3]

    binary = ROI_binary_mask(s)
    contour_binary = GetContours(binary)
    s[ contour_binary > 0] = [0,0,0]

    imsave(os.path.join("/share/data40T_v2/Checking", basename), s)
'''

