from UsefulFunctions.UsefulOpenSlide import GetImage
import numpy as np


def GetLabel(name, para):
	if "Normal" in name:
		return 0
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

    	Label = GetLabel(name, para)
    	
    	para += (file,)
    	para += (Label)
        pre = "{} {} {} {} {} {}\n".format(*para)

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

    (options, args) = parser.parse_args()

    if "Tumor" in options.method:
    	method = "SP_ROI_tumor"
    else:
    	method = "SP_ROI_normal"

    name = options.file

    list_roi = ROI(name, ref_level=0, disk_size=4, thresh=230, black_spots=None,
        number_of_pixels_max=50176, verbose=False, marge=0, method=method,
        contour_size=3, N_squares=(200, 50, 300), seed=None, cut_whitescore=0.8)


    FILENAME = name.replace(".tif", ".txt")
    FILEPATH = os.path.join(options.output_folder, FILENAME)
    CreateFileParam()