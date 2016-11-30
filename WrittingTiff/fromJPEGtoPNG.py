from tifffile import imsave
from scipy.misc import imread
import sys
import os

for i in range(2, len(sys.argv)):
	name_tiff = sys.argv[i].replace('.jpg', '.tiff')
	to_replace = os.path.join(*sys.argv[i].split('/')[0:-1])
	name_tiff = name_tiff.replace(to_replace, sys.argv[1])


	xx = imread(sys.argv[i])
	imsave(name_tiff, xx)
	if i % 1000 == 0:
		print "{} / {}".format(i+1, len(sys.argv))
print "Finished converting {} images".format(len(sys.argv) - 2)
