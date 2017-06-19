from os.path import join
from scipy.misc import imread
from WrittingTiff.Extractors import bin_analyser, PixelSize, MeanIntensity
from optparse import OptionParser

def file_retriever(filename):
    bin_image = join(filename, filename + "_bin.png")
    raw_image = join(filename, filename + "_RAW.png")
    table_n = join(filename, filename + "_table.csv")
    return imread(raw_image), imread(bin_image), table_n




def Extractors():

    list_extract = [PixelSize("Pixel sum", 0), MeanIntensity("Intensity mean 0", 0), 
              MeanIntensity("Intensity mean 5", 5)]

    return list_extract


if __name__ == "__main__":
    

    parser = OptionParser()

    parser.add_option("--folder_name", dest="folder_name",
                      help="Input folder (raw/bin/seg/prob data)")

    (options, args) = parser.parse_args()

    img, bin, table_name= file_retriever(options.folder_name)

    list_extractors = Extractors()

    table = bin_analyser(img, bin, list_extractors, pandas_table=True)
    table.to_csv(table_name) 
    ### write bit to save table in folder
    