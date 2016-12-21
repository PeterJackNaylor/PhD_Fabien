import xml.etree.ElementTree as ET
from UsefulFunctions.RandomUtils import CheckOrCreate
import glob
from scipy.misc import imread, imsave
from skimage.measure import label
import pandas as pd
import pdb
import matplotlib.pylab as plt
import numpy as np

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
    return string, new_string[1], new_string[2]

def load_files():
    """loading the main files"""
    folder = glob.glob('/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/annotations/*.xml')
    data_folder = pd.DataFrame(columns=('Path_XML', 'SlideID', 'CropID'))
    for i, item in enumerate(folder):
        data_folder.loc[i] = parse_name_xml(item)
    data_folder = data_folder.set_index(['SlideID', 'CropID'])

    image = glob.glob("/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/masks/Slide*/*")
    image_folder = pd.DataFrame(columns=('Path_PNG', 'SlideID', 'CropID'))
    for j, item in enumerate(image):
        image_folder.loc[j] = parse_name_png(item)
    image_folder = image_folder.set_index(['SlideID', 'CropID'])

    result = pd.concat([data_folder, image_folder], axis=1)
    return result

def load_png(image_path):
    """returns label image with different intergers for each"""
    return label(imread(image_path))

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

def fuse(xml_pandas, lbl_label):
    """fuses the data from the lbl_label and the xml pandas"""
    res = lbl_label.copy()
    to_remove = list(np.unique(lbl_label))
    to_remove.remove(0)
    for i in xml_pandas.index:
        x = int(xml_pandas.ix[i, "x"])
        y = int(xml_pandas.ix[i, "y"])
        ty = int(xml_pandas.ix[i, "Type"])
        x = max(min(x, 511), 0)
        y = max(min(y, 511), 0)
        val_nuc = lbl_label[y, x] # we have to inverse, maybe due to the xml enconding..
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



def CreateFolder(data_file):
    for string in data_file["Path_PNG"]:
        string = string.replace("masks", "colored_mask")
        string = string.replace(string.split('/')[-1], "")

        print "Creating folder: {}".format(string)
        CheckOrCreate(string)

def SaveName():
    return "/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/colored_mask/Slide{}/CMask_{}.png"
def SaveDiffName():
    return "/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/colored_mask/Slide{}/DiffCMask_{}.png"

def __main__():
    """main function"""
    all_files = load_files()
    CreateFolder(all_files)
    for el in all_files.index:
        xml_path = all_files.ix[el, "Path_XML"]
        png_path = all_files.ix[el, "Path_PNG"]
        img = load_png(png_path)
        xml_data = get_markers(xml_path)
        classed_label = fuse(xml_data, img)
        save_name = SaveName().format(*el)
        diffsave_name = SaveDiffName().format(*el)
        try:
        	diff_png = img.copy()
        	diff_png[diff_png > 0] = 1
        	diff_classed = classed_label.copy()
        	diff_classed[diff_classed > 0] = 1
        	diff = diff_classed - diff_png
        	diff[diff != 0] = 255
        	imsave(diffsave_name, diff)

        	imsave(save_name, classed_label)
        	print "print saved file: {}".format(save_name)
        except:
        	print "Couldn't save file : {}".format(save_name)
        #pdb.set_trace()
