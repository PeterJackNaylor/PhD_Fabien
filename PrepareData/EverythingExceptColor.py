from optparse import OptionParser
import os
import numpy as np
from shutil import copy
import nibabel as ni
from UsefulFunctions.RandomUtils import CheckOrCreate
import glob
from Deprocessing.Morphology import generate_wsl
import skimage.measure as mea
from scipy.misc import imsave
import os
import shutil
from color_split import color_split
import pdb

import matplotlib.pylab as plt

def plot_lbl(img):
    plt.imshow(img, cmap='jet')
    plt.show()


def plot(img):
    plt.imshow(img)
    plt.show()



def GetOptions(verbose=True):

    parser = OptionParser()

# complsory aguments:
    parser.add_option('-i', dest="input", type="string",
                      help="input folder (ToAnnotate for example)")
    parser.add_option("--o_c", dest="output_cellcognition", type="string",
                      help="output cellcognition file")
    parser.add_option("--o_b", dest="output_binary",
                      help="output for the dataset")

    (options, args) = parser.parse_args()
    if verbose:
        print "Input paramters to run:"
        print " \n "

    # complsory aguments:

        print "Input folder         : | {}".format(options.input)
        print "CellCognition folder : | {}".format(options.output_cellcognition)
        print "Binary folder output : | {}".format(options.output_binary)
    CheckOrCreate(options.output_binary)

    return (options, args)

def Map(Slide_number, hidden_number, list_in):
	list_in.append((Slide_number, hidden_number))

def NameCellImg(output, slide_new, i):
	return os.path.join(output, "data", "Slide{}", "Image_{}.png").format(slide_new, i)

def NameCellGT(output, slide_new, i):
	return os.path.join(output, "masks", "Slide{}", "Mask_{}__T00__c00__Z00.png").format(slide_new, i)

def NameBinImg(output, slide_new, i):
	return os.path.join(output, "Slide_{}", "{}_{}.png").format(slide_new, slide_new, i)

def NameBinGT(output, slide_new, i):
	return os.path.join(output, "GT_{}", "{}_{}.png").format(slide_new, slide_new, i)

def FolderCellImg(output, Slide_new):
	path = os.path.join(output, "data", "Slide{}").format(Slide_new)
	print 'Creating {}'.format(path)
	CheckOrCreate(path)

def FolderCellGT(output, Slide_new):
	path = os.path.join(output, "masks", "Slide{}").format(Slide_new)
	print 'Creating {}'.format(path)
	CheckOrCreate(path)

def FolderBinImg(output, Slide_new):	
	path = os.path.join(output, "Slide_{}").format(Slide_new)
	print 'Creating {}'.format(path)
	CheckOrCreate(path)

def FolderGTImg(output, Slide_new):
	path = os.path.join(output, "GT_{}").format(Slide_new)
	print 'Creating {}'.format(path)
	CheckOrCreate(path)

def LoadGT(path):
    image = ni.load(path)
    img = image.get_data()
    if len(img.shape) == 3:
        img = img[:, :, 0].transpose()
    else:
        img = img.transpose()
    return img

def ProcessLabel(path):
    #pdb.set_trace()
    lbl = LoadGT(path)
    lbl = mea.label(lbl)
    lbl = lbl.astype('uint8')
    wsl = generate_wsl(lbl)
    lbl[lbl > 0] = 1
    wsl[wsl > 0] = 1
    lbl = lbl - wsl	
    lbl[lbl > 0] = 255
    return lbl

def copyprint(src, dst):
	print "Copying {} ------> {}".format(src, dst)


if __name__ == '__main__':

	Mapping = {}
	Mapping["141549"] = "02"
	Mapping["160120"] = "07"
	Mapping["162438"] = "06"
	Mapping["498959"] = "04"
	Mapping["508389"] = "08"
	Mapping["536266"] = "09"
	Mapping["544161"] = "10"
	Mapping["572123"] = "03"
	Mapping["574527"] = "11"
	Mapping["581910"] = "01"
	Mapping["588626"] = "05"

	options, args = GetOptions()
	patients = os.path.join(options.input, 'Slide_*')
	patients = glob.glob(patients)


	for patient in patients:
		patients_img = os.path.join(patient, "*"+".png")
		patients_img = glob.glob(patients_img)
		#patients_gt  = patients_img.replace("Slide", "GT")
		for i, patient_img in enumerate(patients_img):
			Slide_id = patient_img.split('/')[-1].split('_')[0]
			patient_gt = patient_img.replace("Slide", "GT").replace(".png", ".nii.gz")
			
			FolderCellImg(options.output_cellcognition, Mapping[Slide_id])
			FolderCellGT(options.output_cellcognition, Mapping[Slide_id])
			FolderBinImg(options.output_binary, Mapping[Slide_id])
			FolderGTImg(options.output_binary, Mapping[Slide_id])
			# Cellcognition first
			patient_img_cellco = NameCellImg(options.output_cellcognition, Mapping[Slide_id], i+1)
			patient_gt_cellco = NameCellGT(options.output_cellcognition, Mapping[Slide_id], i+1)

			# Binary output
			patient_img_bin = NameBinImg(options.output_binary, Mapping[Slide_id], i+1)
			patient_gt_bin = NameBinGT(options.output_binary, Mapping[Slide_id], i+1)

			# copying the img as no modification is needed
			copy(patient_img, patient_img_cellco)
			copy(patient_img, patient_img_bin)
			copyprint(patient_img, patient_img_cellco)
			copyprint(patient_img, patient_img_bin)
			lbl = ProcessLabel(patient_gt)
			imsave(patient_gt_cellco, lbl)
			imsave(patient_gt_bin, lbl)
			copyprint(patient_gt, patient_gt_cellco)
			copyprint(patient_gt, patient_gt_bin)

	in_dir = os.path.join(options.output_cellcognition, "data")
	out_dir = os.path.join(options.output_cellcognition, "decomposed")
	color_split(in_dir, out_dir)




