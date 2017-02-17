"""
python XmlParsing.py --a /Users/naylorpeter/Documents/Histopathologie/CellCognition/classifiers/classifier_January2017/ --c /Users/naylorpeter/Documents/Histopathologie/CellCognition/Fabien/ --o /Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotateColor/ --d /Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotateDiff
"""

import xml.etree.ElementTree as ET
from UsefulFunctions.RandomUtils import CheckOrCreate
import os
import glob
from scipy.misc import imread, imsave
from skimage.measure import label
import pandas as pd
import pdb
import matplotlib.pylab as plt
import numpy as np
from optparse import OptionParser


def plot(img):
	plt.imshow(img)
	plt.show()

def parse_name_xml(string):
    """parsers cellcognitions xml name files"""
    new_string = string.replace('PLSlide', "AZER").replace("___P", "AZER").replace("___T00", "AZER")
    new_string = new_string.split('AZER')
    return string, new_string[1], new_string[2]

def parse_name_png(string):
    """parsers cellcognitions masks name file"""
    new_string = string.replace("/Slide", "AZER").replace("/Mask_", "AZER").replace('.png', 'AZER')
    new_string = new_string.split('AZER')
    return string, new_string[1], new_string[2].split('__')[0]

def load_files(options):
    """loading the main files"""
    files_location = os.path.join(options.annotation, 'annotations/*.xml')
    folder = glob.glob(files_location)
    data_folder = pd.DataFrame(columns=('Path_XML', 'SlideID', 'CropID'))
    for i, item in enumerate(folder):
        data_folder.loc[i] = parse_name_xml(item)
    data_folder = data_folder.set_index(['SlideID', 'CropID'])

    slides_location = os.path.join(options.cellcognition, "masks/Slide*/*.png")
    image = glob.glob(slides_location)
    image_folder = pd.DataFrame(columns=('Path_PNG', 'SlideID', 'CropID'))
    for j, item in enumerate(image):
        image_folder.loc[j] = parse_name_png(item)
    image_folder = image_folder.set_index(['SlideID', 'CropID'])

    result = pd.concat([data_folder, image_folder], axis=1)
    return result

def load_png(image_path):
    """returns label image with different intergers for each"""
    #pdb.set_trace()
    lbl = label(imread(image_path))
    lbl = lbl.astype(np.uint8)
    return lbl

def get_markers(xml_file):
    """ parsers the xml file into a pandas dataframe"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    xml_points = pd.DataFrame(columns=("Type", "x", "y"))
    i = 0
    for neighbor in root.iter('Marker_Type'):
        list_child = neighbor._children
        type_marker = list_child[0]
        type_int = int(type_marker.text)
        list_child.remove(type_marker)
        for marker in list_child:
            x = int(marker._children[0].text)
            y = int(marker._children[1].text)
            xml_points.loc[i] = type_int, x, y
            i += 1
    return xml_points



def encadre(x):
    return max(min(x,511),0)
def find_closest_lbl(img, x, y, thresh=10):
    """ find closest lbl of pixel in a thresh-old array"""
    for i in range(thresh):
        x_min = encadre(x - thresh)
        x_max = encadre(x + thresh)
        y_min = encadre(y - thresh)
        y_max = encadre(y + thresh)
        ana = img[x_min:x_max, y_min:y_max].copy()
        lbl_prox = np.unique(ana)
        if len(lbl_prox) != 1:
            for el in lbl_prox:
                if el != 0:
                    break
            return el
    return 0


def fuse(xml_pandas, lbl_label):
    """fuses the data from the lbl_label and the xml pandas"""
    res = lbl_label.copy()
    to_remove = list(np.unique(lbl_label))
    to_remove.remove(0)
    for i in xml_pandas.index:
        x = int(xml_pandas.ix[i, "x"])
        y = int(xml_pandas.ix[i, "y"])
        ty = int(xml_pandas.ix[i, "Type"])
        x = encadre(x)
        y = encadre(y)
        val_nuc = lbl_label[y, x] # we have to inverse, maybe due to the xml enconding..
        if val_nuc == 0:
            val_nuc = find_closest_lbl(lbl_label, y, x)
        if val_nuc != 0:
            res[lbl_label == val_nuc] = ty
        #print x, y, ty
            if val_nuc in to_remove:
            	to_remove.remove(val_nuc)

    n_pri = len(to_remove)
    for too_small in to_remove:
        res[res == too_small] = 0

    print len(np.unique(lbl_label)), len(np.unique(lbl_label))-1-n_pri, len(xml_pandas.index)

    return res



def CreateFolder(data_file, options):
    for string in data_file["Path_PNG"]:
        string = string.replace(string.split('/')[-1], "")
        ending = string[:-1].split('/')[-1]
        ending = ending.replace('Slide', 'Slide_')
        #pdb.set_trace()
        string = os.path.join(options.output, ending)

        if not os.path.isdir(string):
            print "Creating folder: {}".format(string)
        CheckOrCreate(string)
        GT = string.replace('Slide', "GT")
        CheckOrCreate(GT)
        if not os.path.isdir(GT):
            print "Creating folder: {}".format(GT) 

        if options.output_diff != "None":
            string = os.path.join(options.output_diff, ending)
            if not os.path.isdir(string):
                print "Creating folder: {}".format(string)
            CheckOrCreate(string)
            

def SaveName(options):
    return os.path.join(options.output, "GT_{}/{}_{}.png")
def SaveDiffName(options):
    return os.path.join(options.output_diff, "Slide_{}/{}_{}.png")

def GetOptions(verbose=True):

    parser = OptionParser()

# complsory aguments:
    parser.add_option('--a', dest="annotation", type="string",
                      help="annotation folder")
    parser.add_option("--c", dest="cellcognition", type="string",
                      help="cellcognition output")
    parser.add_option("--o", dest="output", type='string',
                     help="output")
    parser.add_option("--d", dest='output_diff', type="string", default="None",
                     help="images of who hasn't been assigned (if specified)")
    parser.add_option("--b", dest='binary', type="string", 
                     help="binary folder")

    (options, args) = parser.parse_args()
    if verbose:
        print "Input paramters to run:"
        print " \n "

    # complsory aguments:

        print "Annotation folder    : | {}".format(options.annotation)
        print "CellCognition folder : | {}".format(options.cellcognition)
        print "Binary folder        : | {}".format(options.binary)
        print "Output               : | {}".format(options.output)
        print "Output difference    : | {}".format(options.output_diff)
    CheckOrCreate(options.output)
    if options.output_diff != "None":
        CheckOrCreate(options.output_diff)

    return (options, args)


def __main__(options):
    """main function"""
    all_files = load_files(options)
    CreateFolder(all_files, options)
    for el in all_files.index:
        xml_path = all_files.ix[el, "Path_XML"]
        png_path = all_files.ix[el, "Path_PNG"]
        print 'Dealing with {}'.format(png_path)
        print 'And with {}'.format(xml_path)
        img = load_png(png_path)
        #pdb.set_trace()
        xml_data = get_markers(xml_path)
        classed_label = fuse(xml_data, img)
        s_id = el[0]
        i_id = el[1]
        save_name = SaveName(options).format(s_id, s_id, i_id)
        diffsave_name = SaveDiffName(options).format(s_id, s_id, i_id)
        try:
            diff_png = img.copy()
            diff_png[diff_png > 0] = 1
            diff_classed = classed_label.copy()
            diff_classed[diff_classed > 0] = 1
            diff = diff_classed - diff_png
            diff[diff != 0] = 255
            print classed_label.shape
            #pdb.set_trace()
            imsave(diffsave_name, diff)
            imsave(save_name, classed_label)
            print "print saved file: {}".format(save_name)
        except:
            print "Couldn't save file : {}".format(save_name)
        #pdb.set_trace()
if __name__ == "__main__":
    options, args = GetOptions()
    __main__(options)
