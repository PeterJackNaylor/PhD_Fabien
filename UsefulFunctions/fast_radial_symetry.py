import numpy as np
import matplotlib.pylab as plt
from skimage.filters import sobel_h, sobel_v, gaussian
from optparse import OptionParser
from scipy.misc import imread
import pdb

def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    i = 0
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            change = False
            if res_img.shape[0] != windowSize[1]:
                y = image.shape[0] - windowSize[1]
                change = True
            if res_img.shape[1] != windowSize[0]:
                x = image.shape[1] - windowSize[0]
                change = True
            if change:
                res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)

def g(img):
    """
    Computes gradient magnitude at a distance of n
    """
    g_h = sobel_h(img)
    g_v = sobel_v(img)
    return g_v, g_h

def NormAlpha(a, n=2):
    if n==2:
        return (np.dot(a, a))**(0.5)
    else:
        return np.linalg.norm(a, n)
def NormAlphaImage(img, a):
    f = np.vectorize(NormAlpha())
    return f(img)

def closest_int(float_n):
    return int(round(float_n))

def PpAndNp(p, g_p, n):
    """

    """

    direction = g_p / NormAlpha(g_p, 2)
    shift = map(closest_int, n * direction)
    return p + shift, p - shift

def S(original, list_n, alpha = 2, sigma = None):
    """
    input:
        - img, img in numpy array format
        - n range of transformation
        - sigma covariance matrix
        - alpha: radial strickness parameter
    """
    img = original.copy()
    offset = np.max(list_n)
    dim = img.shape
    grad_x, grad_y = g(img)

    On = np.zeros(shape=(dim[0] + 2 * offset, dim[1] + 2 * offset)).astype(float)
    Mn = np.zeros(shape=(dim[0] + 2 * offset, dim[1] + 2 * offset)).astype(float)
    res = np.zeros_like(On)
    res = np.zeros(shape=(dim[0] + 2 * offset, dim[1] + 2 * offset)).astype(float)
    for n in list_n:
        if sigma is None:
            sigma = n * 0.5
        On = np.zeros_like(res)
        Mn = np.zeros_like(res)
        for i in range(dim[0]):
            for j in range(dim[1]):
                p = np.array([i, j])
                grad_p = np.array([ grad_x[p[0], p[1]], grad_y[p[0], p[1]]])
                
                g_norm = NormAlpha(grad_p, 2)
                if g_norm > 0:
                    direction = map(closest_int, grad_p / g_norm)
                    ppve = p + direction + offset

                    On[ppve[0], ppve[1]] += 1
                    Mn[ppve[0], ppve[1]] += g_norm

                    pnve = p - direction + offset

                    On[pnve[0], pnve[1]] -= 1
                    Mn[pnve[0], pnve[1]] -= g_norm
        On = On / np.max(On)
        Mn = Mn / np.max(Mn)

        res += gaussian((On**(alpha)) * Mn, sigma)

    return res[offset:-offset, offset:-offset]


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input",
                      help="Input file (raw data)")
    parser.add_option("-n",  dest="n", type="int",
                      help="window size")
    parser.add_option('--sigma', dest="sigma", type="int",
                      help="value of sigma")
    (options, args) = parser.parse_args()

    input_img = imread(options.input)
    output_img = S(input_img, range(20), sigma=options.sigma)

