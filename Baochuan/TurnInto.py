import glob as g
from skimage.io import imread
import nibabel as nib
import numpy as np


files_GT = g.glob('/home/naylor/Bureau/BaochuanPang/GT_*/*.tif')
dt = np.dtype(zip('RGB', ('u1',) * 3))

for files in files_GT:
    files_nii = files.replace('.tif', '.nii.gz')

    imdata = imread(files)
    imdata = np.transpose(imdata, (1, 0))
    nii = nib.Nifti1Image(imdata, np.eye(4))
    print "writting %s" % files_nii
    nii.to_filename(files_nii)
