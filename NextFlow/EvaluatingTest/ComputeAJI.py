from optparse import OptionParser
from scipy.misc import imread
from os.path import join
import pdb
from Prediction.AJI import AJI_fast


def Options():

    parser = OptionParser()

    parser.add_option('--fold', dest="fold", type="str",
                      help="folder to analyse")
    (options, args) = parser.parse_args()
    return options




if __name__ == '__main__':
    options = Options()

    anno_path = join(options.fold, "Label.png")
    PP_path = join(options.fold, "Bin.png")
    file_name = join(options.fold, "Characteristics.txt")

    anno = imread(anno_path)
    PP = imread(PP_path)

    score = AJI_fast(anno, PP)

    f = open(file_name, 'a')
    f.write('AJI: # {} #\n'.format(score))
    f.close()
