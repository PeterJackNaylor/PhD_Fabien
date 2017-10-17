from UsefulFunctions.CuttingPatches import ROI
from optparse import OptionParser
import os

def CreateFileParam(name, list, slidename):
    """
    Creates physically a text file named name where each line as an id 
    and each line as parameters
    """
    f = open(name, "wb")
    line = 1
    for para in list:
        pre = "__{}__ ".format(line)
        pre += "{} {} {} {} {}".format(*para)
        pre += " {}".format(slidename)
        pre += "\n"
        f.write(pre)
        line += 1
    f.close()

if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--slide", dest="slide",type="string",
                      help="slide name")
    parser.add_option("--output", dest="out",type="string",
                      help="out path")
    parser.add_option("--method", dest="method",type="string",
                      help="method to use")
    parser.add_option("--marge", dest="marge", type="int",
              help="how much to reduce indexing")

    (options, args) = parser.parse_args()




    list_of_para = ROI(options.slide, ref_level=0, disk_size=4, thresh=230, 
               black_spots=None, number_of_pixels_max=9000000, 
               verbose=False, marge=options.marge, method=options.method, 
               mask_address=None, contour_size=3, N_squares=100, 
               seed=None, fixed_size_in=(512, 512), fixed_size_out=(512,512))
    CreateFileParam(options.out, list_of_para, os.path.basename(options.slide).replace(".tiff", ""))
