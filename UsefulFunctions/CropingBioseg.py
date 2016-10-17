from PIL import Image
import glob as g
import numpy as np
import pdb


def GetNames(directory):
    all_files = g.glob(directory + "/*.tif")
    return all_files


def LoadImgLbl(file, final_size=224):

    rgb = image2pixelarray(file)
    lbl = image2pixelarray(file.replace('Slide', 'GT').replace(
        '_ccd', '').replace('.tif', '.TIF'))

    x, y = np.where(lbl > 0)
    x_g = np.min(x)
    x_d = np.max(x)
    y_h = np.min(y)
    y_b = np.max(y)
    # pdb.set_trace()
    x_g, x_d = ExtractBounderie(x_g, x_d, (0, lbl.shape[0]), final_size)
    y_h, y_b = ExtractBounderie(y_h, y_b, (0, lbl.shape[1]), final_size)

    return rgb[x_g:x_d, y_h:y_b], lbl[x_g:x_d, y_h:y_b]


def ExtractBounderie(x_1, x_2, image_bounderies, final_size, always_crop_to_size=True):
    # pdb.set_trace()
    if not x_2 > x_1:
        raise Exception("x_2 must be superior to x_1")
    if x_2 - x_1 > final_size:
        if always_crop_to_size:
            val = (x_2 - x_1 - final_size) / 2
            x_2 -= val
            x_1 += val
        else:
            raise Exception(
                "points are too far appart to restrict them to the size of %d" % final_size)
    val = (final_size - (x_2 - x_1)) / 2

    space_to_the_right = image_bounderies[1] - x_2
    space_to_the_left = x_1 - image_bounderies[0]

    if space_to_the_left > space_to_the_right:
        if space_to_the_right > val:
            x_1 = x_1 - val
            x_2 = x_2 + val
        else:
            x_1, x_2 = image_bounderies[1] - final_size, image_bounderies[1]
    else:
        if space_to_the_left > val:
            x_1, x_2 = x_1 - val, x_2 + val
        else:
            x_1, x_2 = image_bounderies[0], final_size + image_bounderies[0]
    if x_2 - x_1 < final_size:
        to_add = final_size - x_2 + x_1

        if x_1 - to_add >= image_bounderies[0]:
            x_1 = x_1 - to_add
        else:
            x_2 += to_add
    return x_1, x_2


def image2pixelarray(filepath):
    im = Image.open(filepath).convert('L')
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map).reshape((height, width))
    return greyscale_map


if __name__ == "__main__":
    # import matplotlib.pylab as plt
    from scipy.misc import imsave
    import os
    files = GetNames("/home/naylor/Bureau/NewDataSet/Slide")
    res_fold_slide = "/home/naylor/Bureau/NewSlides/Slide_111111"
    res_fold_gt = "/home/naylor/Bureau/NewSlides/GT_111111"
    size = 224
    for f in files:
        # fig, axes = plt.subplots(1, 2)
        rgb, lbl = LoadImgLbl(f, size)
        name = f.split('/')[-1]
        print f + "   :"
        assert rgb.shape[:2] == lbl.shape[:2]
        print rgb.shape
        imsave(os.path.join(res_fold_slide, name), rgb)
        imsave(os.path.join(res_fold_gt, name), lbl)
