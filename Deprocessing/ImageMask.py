from sklearn.cluster import KMeans
from scipy.misc import imread
import numpy as np


def flatten_image(image):
	x, y, z = image.shape
	result = np.zeros(shape=(x*y, z))
	for i in range(z):
		result[:,i] = image[:,:,i].flatten('C')
	return result


def GetSegmentation(image):
	image_flat = flatten_image(image)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(image_flat)
	output = kmeans.labels_.reshape(image.shape[0:2])
	return output



if __name__ == '__main__':
	image = imread('/Users/naylorpeter/Desktop/doublons/483552.png')
	segmented = GetSegmentation(image)

