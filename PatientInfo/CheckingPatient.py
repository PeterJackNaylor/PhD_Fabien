from optparse import OptionParser
from os.path import join
from UsefulFunctions.UsefulOpenSlide import GetWholeImage
from scipy.misc import imsave
from UsefulFunctions.RandomUtils import CheckOrCreate
from glob import glob

def GetOptions(verbose=True):

    parser = OptionParser()
    parser.add_option('--i', dest="input", type="string",
                      help="input folder")
    parser.add_option("--o", dest="output", type='string',
                     help="output folder")
    (options, args) = parser.parse_args()

    if verbose:
        print " \n "
        print "Input paramters to run:"
        print " \n "
        print "Input folder         : | {}".format(options.input)
        print "Output               : | {}".format(options.output)

    CheckOrCreate(options.output)

    return (options, args)


def __main__(options):
    output = options.output
    input = options.input
    input_files = join(input, '*.tiff')
    for f in glob(input_files):
        image = GetWholeImage(f)
        savename = join(output, f.split("/")[-1])
        print "Saving file: {}".format(savename)
        imsave(savename, image)


if __name__ == "__main__":
    options, args = GetOptions()
    __main__(options)
