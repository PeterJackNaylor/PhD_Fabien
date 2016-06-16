import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp

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
    res = cv2.flip(picture,1)
    return res
    
def flip_horizontal(picture):
    res = cv2.flip(picture,0)
    return res

class Transf(object):
    
    def __init__(self,name):
        self.name = name
        
    def _apply_(self, image):
        raise NotImplementedError
        
    def enlarge(self,image, x, y):
        
        rows, cols, channels = image.shape
        
        enlarged_image = np.zeros(shape=(rows + 2 * y, cols + 2 * x, channels))
        
        enlarged_image[y:(y+rows),x:(x+cols), 0:channels] = image
        
        #top part:
        enlarged_image[0:y, x:(x+cols), 0:channels] = flip_horizontal(enlarged_image[y:(2 * y), x:(x+cols),0:channels])
        
        #bottom part:
        enlarged_image[(y+rows):(2 * y + rows), x:(x+cols),0:channels] = flip_horizontal(enlarged_image[rows:(y+rows), x:(x+cols),0:channels])

        #left part:
        enlarged_image[y:(y+rows), 0:x, 0:channels] = flip_vertical(enlarged_image[y:(y+rows), x:(2 * x),0:channels])
        
        #right part:
        enlarged_image[y:(y+rows), (cols + x):(2 * x + cols), 0:channels] = flip_vertical(enlarged_image[y:(y+rows), cols:(cols + x),0:channels])

        #top left from left part:
        enlarged_image[0:y, 0:x, 0:channels] = flip_horizontal(enlarged_image[y:(2 * y), 0:x, 0:channels])
        
        #top right from right part:
        enlarged_image[0:y, (x + cols):(2 * x + cols), 0:channels] = flip_horizontal(enlarged_image[y:(2 * y), cols:(x + cols), 0:channels])
        
        #bottom left from left part:
        enlarged_image[(y+rows):(2 * y + rows), 0:x, 0:channels] = flip_horizontal(enlarged_image[rows:(y + rows), 0:x, 0:channels])
        
        #bottom right from right part
        enlarged_image[(y+rows):(2 * y + rows), (x + cols):(2 * x + cols), 0:channels] = flip_horizontal(enlarged_image[rows:(y + rows), (x + cols):(2 * x + cols), 0:channels])
        enlarged_image = enlarged_image.astype('uint8')
        return(enlarged_image)
class Identity(Transf):

    def __init__(self):

        Transf.__init__(self,"identity")

    def _apply_(self, image):

        return image
        
        

class Translation(Transf):
    
    def __init__(self, x, y, enlarge = True):
    
        Transf.__init__(self, "Trans_" + str(x) + "_" + str(y))
        if x < 0 :
            x = - x
            x_rev = -1
        else:
            x_rev =  1
        if y < 0 :
            y = - y
            y_rev = -1
        else:
            y_rev = 1
        
        self.params = {"x" : x, "y" : y, "rev_x": x_rev, "rev_y": y_rev, "enlarge": enlarge}
    
    def _apply_(self, image):
        
        rows, cols, channels = image.shape

        x = self.params['x']
        y = self.params['y']
        rev_x = self.params['rev_x']
        rev_y = self.params['rev_y']
        enlarge = self.params['enlarge']
        
        if enlarge:
            big_image = self.enlarge(image, x, y)
            res = big_image[(x + rev_x*x):(rows + x + rev_x*x), (y + rev_y*y):(cols + y + rev_y*y),:]
        else:
            M = np.float32([[1,0,(x * rev_x * -1)],[0,1,(y * rev_y * -1)]])
            res = cv2.warpAffine(image, M, (cols,rows))
        return res
        
        
class Rotation(Transf):
    
    def __init__(self, deg, enlarge = True):

        Transf.__init__(self, "Rot_" + str(deg))
        self.params = {"deg" : deg, "enlarge" : enlarge}
    
    def _apply_(self, image):
        
        rows, cols, channels = image.shape

        deg = self.params['deg']
        enlarge = self.params['enlarge']
        
        if enlarge:
            ### this part could be better adjusted
            x = int(rows * (2 - 1.414213)/1.414213) 
            y = int(cols * (2 - 1.414213)/1.414213)

            z = max(x, y)
            big_image = self.enlarge(image, z, z)

            b_rows, b_cols, b_channels = big_image.shape
            M = cv2.getRotationMatrix2D((b_cols/2,b_rows/2),deg,1)
            dst = cv2.warpAffine(big_image,M,(b_cols,b_rows))
            if b_channels == 1:
                dst_ = np.zeros(shape=(b_rows, b_cols, b_channels))
                dst_[:,:,0] = dst
                dst = dst_.copy()
                del dst_
            res = dst[z:(z+rows),z:(z+cols),:]
        else:
            M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
            res = cv2.warpAffine(image, M, (cols,rows))
        return res

class Flip(Transf):
    
    def __init__(self, hori):
        if hori != 0 and hori!= 1:
            print "you must give a integer, your parameter is ignored"
            hori = 1
        Transf.__init__(self, "Flip_" + str(hori))
        self.params = {"hori" : hori}
    
    def _apply_(self, image):
        
        hori = self.params["hori"]
        
        if hori == 1:
            res = flip_horizontal(image)
        else:
            res = flip_vertical(image)
        
        return res


class OutOfFocus(Transf):
    
    def __init__(self, sigma):
        Transf.__init__(self, "OutOfFocus_" + str(sigma))
        self.params = {"sigma" : sigma}
    
    def _apply_(self, image):
        
        sigma = self.params["sigma"]
        
        res = cv2.blur(image, (sigma, sigma))
    
        return res


class ElasticDeformation(Transf):
    def __init__(self, mu, sigma, num_points):
        Transf.__init__(self, "ElasticDeform_" + str(mu) + "_" + str(sigma) + "_" + str(num_points))
        self.params = { "mu" : mu, "sigma" : sigma, "num_points": num_points}
        
    def grid(self, rows, cols, num_points):
        ### returns a grid in the form of a stacked array x is 0 and y is 1
        src_cols = np.linspace(0, cols, num_points)
        src_rows = np.linspace(0, rows, num_points)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        return src

    def _apply_(self, image):

        mu = self.params["mu"]
        sigma = self.params["sigma"]
        num_points = self.params["num_points"]
                
        res = image.copy()
        rows, cols = image.shape[0], image.shape[1]

        src = self.grid(rows, cols, num_points)
        # add gaussian displacement to row coordinates
        dst_rows = src[:, 1] - sigma * np.random.randn(src.shape[0]) + mu
        dst_cols = src[:, 0] - sigma * np.random.randn(src.shape[0]) + mu

        ## Delimiting points to the grid space
        for point_ind in range(src.shape[0]):
            dst_rows[point_ind] = min(max(dst_rows[point_ind],0),rows)
            dst_cols[point_ind] = min(max(dst_cols[point_ind],0),cols)


        dst = np.vstack([dst_cols, dst_rows]).T


        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)

        out_rows = rows
        out_cols = cols
        res = warp(image, tform, output_shape=(out_rows, out_cols))
        
        return res
