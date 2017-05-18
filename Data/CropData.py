import glob as g
from os.path import join 
from skimage import measure
import numpy as np
from scipy import misc

INPUT = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotateColor/"

OUTPUT = "/Users/naylorpeter/Desktop/Crop"

count = 0
txt_lbl = open(join(OUTPUT, 'label.txt'),'w') 

crop_box_size = 40

def crop(IMAGE, X, Y, SIZE):
    
    dim = IMAGE.shape
    x_min, x_max = X - SIZE / 2, X + SIZE / 2
    y_min, y_max = Y - SIZE / 2, Y + SIZE / 2

    if y_min < 0: #if too low
        y_max -= y_min # because y_min is negative
        y_min = 0
    
    if y_max > dim[1]: #if too high
        y_min -= y_max - dim[1]
        y_max = dim[1]

    if x_min < 0: #if too low
        x_max -= x_min # because y_min is negative
        x_min = 0
    
    if x_max > dim[1]: #if too high
        x_min -= x_max - dim[0]
        x_max = dim[0]

    return IMAGE[x_min:x_max, y_min:y_max]



for slide_ in g.glob(join(INPUT, "Slide_*/*.png")):
    img = misc.imread(slide_)
    lbl = misc.imread(slide_.replace("Slide", "GT"))
    labeled = measure.label(lbl, background = 0)

    for i in range(1, np.max(labeled)):
        x, y = np.where(labeled == i)
        x_med = int( (np.max(x) + np.min(x)) / 2 )
        y_med = int( (np.max(y) + np.min(y)) / 2 )

        number_label = lbl[x_med, y_med]
        sub_img = crop(img, x_med, y_med, crop_box_size)
        BASENAME = "{:04d}.png".format(count)
        txt_lbl.write(BASENAME + " {}".format(number_label) + "\n")
        filename = join(OUTPUT, BASENAME)
        misc.imsave(filename, sub_img)
        count += 1