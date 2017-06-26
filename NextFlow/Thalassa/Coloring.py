from skimage.morphology import reconstruction, dilation, disk
import cv

def AndImg(a, b):
	## bin images
	if np.max(a) > 1 or np.max(b) > 1:
		raise NotImplementedError
	x, y = np.where(b == 1)
	tmp = a.copy()
	tmp[x, y] += 1
	tmp[tmp != 2] = 0
	tmp[tmp > 0] = 1
	return tmp

def find_closest_cc(x, y, b):
	copy = np.zeros_like(b)
	copy[x, y] = 1
	dilat = 1
	found = False
	while not found:
		dilatedd = dilation(copy, disk(dilat))
		inter = cv.bitwise_and(b, dilatedd)
		if np.sum(inter) != 0:
			found = True
		else:
			dilat += 1
	xx, yy = np.where(inter == 1)
	return xx[0], yy[0]

def change_color(table, bin, color_vec, indice, res):
	x_cent, y_cent = table[indice, 3:5]
	X, Y = int(x_cent), int(y_cell)
	only_cell = np.zeros_like(bin)
	if bin[X, Y] != 1:
		X, Y = find_closest_cc(X, Y, bin)


	only_cell[X, Y] = 1
	only_cell = reconstruction(only_cell, bin)
	x, y = np.where(only_cell == 1)
	res[x, y] = color_vec[indice]

def f
