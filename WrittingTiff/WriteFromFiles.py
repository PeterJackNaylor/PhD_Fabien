#### for this programme to work, you have to set the value of python open files much higher
### sudo bash
### export LD_LIBRARY_PATH
### run setlimit.py to raise the number and voila!

import pdb
import sys
import openslide
from gi.repository import Vips

if len(sys.argv) < 4:
    print "usage: %s n image-out image1 image2 ..." % sys.argv[0]
    print "   make an n x n grid of images"
    sys.exit(1)

slide_name = sys.argv[1]
size_x, size_y = openslide.open_slide(slide_name).dimensions

outfile = sys.argv[2]


img = Vips.Image.black(size_x, size_y)
val = min(len(sys.argv), 1000000)
for i in range(3, val):
	if i % 1000 == 0:
		print "{} / {}".format(i, len(sys.argv))
	tile = Vips.Image.new_from_file(sys.argv[i], 
			                    access = Vips.Access.SEQUENTIAL_UNBUFFERED)
	_x, _y = sys.argv[i].split('.')[0].split('/')[-1].split('_')
	img = img.insert(tile, int(_x), int(_y))

img.tiffsave(outfile, tile=True, pyramid=True, bigtiff = True)