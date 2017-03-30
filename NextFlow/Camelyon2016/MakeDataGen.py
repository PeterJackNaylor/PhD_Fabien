

from optparse import OptionParser
from Data.DataGenScratch import DataGenScratch

from Training.options import AddTransforms
import UsefulFunctions.ImageTransf as T
import cPickle as pkl
import pdb

def MakeDataGen(options):
    dg_output = options.output

    data_generator= DataGenScratch(options.input, options.split,  
                                          options.transform_list, size = None, random_crop = False, Unet = False,
                                          img_format = "RGB", seed=1234, crop=1, name="CAM16", pathfolder=options.pathfolder)
    pkl.dump(data_generator, open(dg_output, "wb"))


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input",
                      help="input text folder")

    parser.add_option("-o", "--output", dest="output",
                      help="Where to store the pkl file")
    parser.add_option("--split", dest="split",
                      help="What kind of split to do")
    parser.add_option("--pathfolder", dest="pathfolder",
                      help="path to the image cutten data")
    (options, args) = parser.parse_args()
    options.enlarge = True
    if options.split == 'train':
        #options = AddTransforms(options)
        options.transform_list = [T.Identity()]
    else:
        options.transform_list = [T.Identity()]
    MakeDataGen(options)

