import nibabel as ni
import pdb
from skimage.io import imread, imsave
from glob import glob
from os.path import dirname, join, basename
from shutil import copy
from UsefulFunctions.RandomUtils import CheckOrCreate
from scipy.ndimage.morphology import distance_transform_cdt
from skimage.measure import label, regionprops
from Deprocessing.Morphology import generate_wsl
import pdb

def LoadGT(path):

    image = ni.load(path)
    img = image.get_data()
    img = label(img)
    wsl = generate_wsl(img[:,:,0])
    img[ img > 0 ] = 1
    wsl[ wsl > 0 ] = 1
    img[:,:,0] = img[:,:,0] - wsl
    if len(img.shape) == 3:
        img = img[:, :, 0].transpose()
    else:
        img = img.transpose()
    return img


NEW_FOLDER = 'ToAnnotateDistance'
CheckOrCreate(NEW_FOLDER)
for image in glob('ToAnnotate/Slide_*/*.png'):
    baseN = basename(image)
    Slide_name = dirname(image)
    GT_name = baseN.replace('.png', '.nii.gz')
    OLD_FOLDER = dirname(Slide_name)
    Slide_N = basename(dirname(image))
    GT_N = Slide_N.replace('Slide_', 'GT_')
    
    CheckOrCreate(join(NEW_FOLDER, Slide_N))
    CheckOrCreate(join(NEW_FOLDER, GT_N))

    copy(image, join(NEW_FOLDER, Slide_N, baseN))
    bin_image = LoadGT(join(OLD_FOLDER, GT_N, GT_name))
    res = distance_transform_cdt(bin_image)
    props = regionprops(label(bin_image))
    res = res.astype('float')
    for p in props:
        MAX_VALUE = res[p.coords].max()
        for x, y in p.coords:
            res[x, y] = res[x, y] / float(MAX_VALUE)
    res = res * 255
    res = res.astype('uint8')
    imsave(join(NEW_FOLDER, GT_N, baseN), res)