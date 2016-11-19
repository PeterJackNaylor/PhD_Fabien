import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import img_as_ubyte
import pdb
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#==============================================================================
#
# def flip_vertical(picture):
#     result = picture.copy()
#     height, width , channel= result.shape
#
#     for x in range(0, width/2):   # Only process the half way
#         for y in range(0, height):
#         # swap pix and pix2
#             #print [y, width  - x]
#             result[y, width  - x - 1,  :] = picture[y, x, :]
#             result[y, x, :] = picture[y, width - x - 1, :]
#     return result
#
# def flip_horizontal(picture):
#     result = picture.copy()
#     height, width , channel= result.shape
#
#     for y in range(0, height/2):   # Only process the half way
#         for x in range(0, width):
#         # swap pix and pix2
#             #print [y, width  - x]
#             result[y, x,  :] = picture[height - 1 - y, x, :]
#             result[height - 1 - y, x, :] = picture[y, x, :]
#     return result
#
#==============================================================================


def flip_vertical(picture):
    res = cv2.flip(picture, 1)
    return res


def flip_horizontal(picture):
    res = cv2.flip(picture, 0)
    return res


class Transf(object):

    def __init__(self, name):
        self.name = name

    def _apply_(self, *image):
        raise NotImplementedError

    def enlarge(self, image, x, y):

        rows, cols = image.shape[0], image.shape[1]
        if len(image.shape) == 2:
            enlarged_image = np.zeros(shape=(rows + 2 * y, cols + 2 * x))
        else:
            enlarged_image = np.zeros(shape=(rows + 2 * y, cols + 2 * x, 3))

        enlarged_image[y:(y + rows), x:(x + cols)] = image

        # top part:
        enlarged_image[0:y, x:(x + cols)] = flip_horizontal(
            enlarged_image[y:(2 * y), x:(x + cols)])

        # bottom part:
        enlarged_image[(y + rows):(2 * y + rows), x:(x + cols)] = flip_horizontal(
            enlarged_image[rows:(y + rows), x:(x + cols)])

        # left part:
        enlarged_image[y:(y + rows), 0:x] = flip_vertical(
            enlarged_image[y:(y + rows), x:(2 * x)])

        # right part:
        enlarged_image[y:(y + rows), (cols + x):(2 * x + cols)] = flip_vertical(
            enlarged_image[y:(y + rows), cols:(cols + x)])

        # top left from left part:
        enlarged_image[0:y, 0:x] = flip_horizontal(
            enlarged_image[y:(2 * y), 0:x])

        # top right from right part:
        enlarged_image[0:y, (x + cols):(2 * x + cols)] = flip_horizontal(
            enlarged_image[y:(2 * y), cols:(x + cols)])

        # bottom left from left part:
        enlarged_image[(y + rows):(2 * y + rows), 0:x] = flip_horizontal(
            enlarged_image[rows:(y + rows), 0:x])

        # bottom right from right part
        enlarged_image[(y + rows):(2 * y + rows), (x + cols):(2 * x + cols)] = flip_horizontal(
            enlarged_image[rows:(y + rows), (x + cols):(2 * x + cols)])
        enlarged_image = enlarged_image.astype('uint8')

        return(enlarged_image)

    def SetFlag(self, i):
        if i == 0:
            return cv2.INTER_LINEAR
        elif i == 1:
            return cv2.INTER_NEAREST
        elif i == 2:
            return cv2.INTER_LINEAR

    def OutputType(self, image):
        # pdb.set_trace()
        if image.dtype == "uint8":
            return image
        elif image.dtype == "uint16":
            cvuint8 = cv2.convertScaleAbs(image)
            return cvuint8
        else:
            image = img_as_ubyte(image)
            return image


class Identity(Transf):

    def __init__(self):

        Transf.__init__(self, "identity")

    def _apply_(self, *image):
        res = ()
        for img in image:
            res += (self.OutputType(img), )
        return image


class Translation(Transf):

    def __init__(self, x, y, enlarge=True):

        Transf.__init__(self, "Trans_" + str(x) + "_" + str(y))
        if x < 0:
            x = - x
            x_rev = -1
        else:
            x_rev = 1
        if y < 0:
            y = - y
            y_rev = -1
        else:
            y_rev = 1

        self.params = {"x": x, "y": y, "rev_x": x_rev,
                       "rev_y": y_rev, "enlarge": enlarge}

    def _apply_(self, *image):

        rows, cols, channels = image.shape

        x = self.params['x']
        y = self.params['y']
        rev_x = self.params['rev_x']
        rev_y = self.params['rev_y']
        enlarge = self.params['enlarge']
        res = ()
        n_img = 0
        for img in image:
            flags = self.SetFlag(n_img)
            if enlarge:
                big_image = self.enlarge(img, x, y)
                res += (big_image[(x + rev_x * x):(rows + x + rev_x * x),
                                  (y + rev_y * y):(cols + y + rev_y * y), :], )
            else:
                M = np.float32([[1, 0, (x * rev_x * -1)],
                                [0, 1, (y * rev_y * -1)]])
                res += (self.OutputType(cv2.warpAffine(image,
                                                       M, (cols, rows)), flags=flags), )
            n_img += 1
            return res


class Rotation(Transf):

    def __init__(self, deg, enlarge=True):

        Transf.__init__(self, "Rot_" + str(deg))
        self.params = {"deg": deg, "enlarge": enlarge}

    def _apply_(self, *image):

        deg = self.params['deg']
        enlarge = self.params['enlarge']
        res = ()
        n_img = 0
        for img in image:
            rows, cols = img.shape[0], img.shape[1]
            flags = self.SetFlag(n_img)
            if enlarge:
                # this part could be better adjusted
                x = int(rows * (2 - 1.414213) / 1.414213)
                y = int(cols * (2 - 1.414213) / 1.414213)

                z = max(x, y)
                big_image = self.enlarge(img, z, z)

                b_rows, b_cols = big_image.shape[0], big_image.shape[1]
                M = cv2.getRotationMatrix2D((b_cols / 2, b_rows / 2), deg, 1)
                dst = cv2.warpAffine(big_image, M, (b_cols, b_rows), flags=flags)

                res += (self.OutputType(dst[z:(z + rows), z:(z + cols)]), )
            else:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
                sub_res = cv2.warpAffine(img, M, (cols, rows), flags=flags)
                res += (self.OutputType(sub_res),)
            n_img += 1

        return res


class Flip(Transf):

    def __init__(self, hori):
        if hori != 0 and hori != 1:
            print "you must give a integer, your parameter is ignored"
            hori = 1
        Transf.__init__(self, "Flip_" + str(hori))
        self.params = {"hori": hori}

    def _apply_(self, *image):
        res = ()
        for img in image:
            hori = self.params["hori"]
            if hori == 1:
                sub_res = flip_horizontal(img)
            else:
                sub_res = flip_vertical(img)

            res += (self.OutputType(sub_res),)
        return res


class OutOfFocus(Transf):

    def __init__(self, sigma):
        Transf.__init__(self, "OutOfFocus_" + str(sigma))
        self.params = {"sigma": sigma}

    def _apply_(self, *image):
        res = ()
        n_img = 0
        for img in image:
            if n_img == 1:
                sub_res = self.OutputType(img)
            else:
                sigma = self.params["sigma"]

                sub_res = cv2.blur(img, (sigma, sigma))

                sub_res = self.OutputType(sub_res)
            n_img += 1
            res += (sub_res,)
        return res


class ElasticDeformation(Transf):

    def __init__(self, alpha, sigma, alpha_affine, seed=None):
        Transf.__init__(self, "ElasticDeform_" + str(alpha) +
                        "_" + str(sigma) + "_" + str(alpha_affine))
        self.params = {"sigma": sigma,
                       "alpha": alpha, "alpha_affine": alpha_affine}

        self.seed = seed

    def grid(self, rows, cols, num_points):
        # returns a grid in the form of a stacked array x is 0 and y is 1
        src_cols = np.linspace(0, cols, num_points)
        src_rows = np.linspace(0, rows, num_points)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        return src

    def _apply_(self, *image):
        res = ()
        n_img = 0
        for img in image:
            shape = img.shape
            shape_size = shape[:2]
            if not hasattr(self, "M"):
                alpha = img.shape[1] * self.params["alpha"]
                alpha_affine = img.shape[1] * self.params["alpha_affine"]
                sigma = img.shape[1] * self.params["sigma"]
                # Random affine
                center_square = np.float32(shape_size) // 2
                square_size = min(shape_size) // 3
                random_state = np.random.RandomState(None)

                pts1 = np.float32([center_square + square_size, [center_square[
                                  0] + square_size, center_square[1] - square_size], center_square - square_size])
                pts2 = pts1 + \
                    random_state.uniform(-alpha_affine, alpha_affine,
                                         size=pts1.shape).astype(np.float32)
                self.M = cv2.getAffineTransform(pts1, pts2)
                self.dx = gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), sigma) * alpha
                self.dy = gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), sigma) * alpha
                # self.dz = np.zeros_like(self.dx)

            if len(shape) == 3:
                x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(
                    shape[0]), np.arange(shape[2]), indexing='ij')
                indices = np.reshape(
                    x + self.dx, (-1, 1)), np.reshape(y + self.dy, (-1, 1)), np.reshape(z, (-1, 1))
            elif len(shape) == 2:
                x, y = np.meshgrid(np.arange(shape[1]), np.arange(
                    shape[0]), indexing='ij')
                indices = np.reshape(
                    x + np.mean(self.dx, axis=2), (-1, 1)), np.reshape(y + np.mean(self.dy, axis=2), (-1, 1))
            else:
                print "Error"

            if n_img == 1:
                order = 0
                flags = cv2.INTER_NEAREST
            else:
                order = 1
                flags = cv2.INTER_LINEAR
            img = cv2.warpAffine(img, self.M, shape_size[
                                 ::-1], flags=flags, borderMode=cv2.BORDER_REFLECT_101)

            sub_res = map_coordinates(
                img, indices, order=order, mode='reflect').reshape(shape)
            sub_res = self.OutputType(sub_res)
            res += (sub_res,)
            n_img += 1
        return res
