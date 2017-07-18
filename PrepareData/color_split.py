import skimage
import skimage.io

import os
import pdb


def color_split(infolder, outfolder):
    for slide in [el for el in os.listdir(infolder) if 'Slide' in el]:
        if slide == ".DS_Store":
            print "bad.."
        else:
            slide_dir = os.path.join(infolder, slide)
            filenames = filter(lambda x: os.path.splitext(
                x)[-1].lower() == '.png', os.listdir(slide_dir))
            OUT_DIR_slide = os.path.join(outfolder, slide)
            if not os.path.isdir(outfolder):
                os.makedirs(outfolder)
            if not os.path.isdir(OUT_DIR_slide):
                os.makedirs(OUT_DIR_slide)
            for filename in filenames:
                full_filename = os.path.join(slide_dir, filename)
                extension = os.path.splitext(full_filename)[-1]
                img = skimage.io.imread(full_filename)

                for i in range(3):
                    out_filename = os.path.join(OUT_DIR_slide, filename.replace(
                        extension, '__c%02i%s' % (i, extension)))
                    print "{} channel {} is saved under {}".format(filename, i+1, out_filename)
                    skimage.io.imsave(out_filename, img[:, :, i])

            #^ % (plate)s$/ ^ % (pos)s$/ .*P % (pos)s_T % (time)05d_C % (channel)s_Z % (zslice)d_S1.tif
if __name__ == '__main__':
    IN_DIR = '/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/data'
    OUT_DIR = '/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/decomposed'

    color_split(IN_DIR, OUT_DIR)