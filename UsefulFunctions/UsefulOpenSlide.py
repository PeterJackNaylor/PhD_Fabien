# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""


import openslide
import numpy as np
import pdb

def GetImage(c, para):
    ## Returns cropped image given a set of parameters
    if len(para)!=5:
            print "Not enough parameters..."
    elif isinstance(c,str):
        sample=openslide.open_slide(c).read_region((para[0],para[1]),para[4],(para[2],para[3]))
    else:
        sample=c.read_region((para[0],para[1]),para[4],(para[2],para[3]))

    return(sample)
    
def GetWholeImage(c, level =None):
    
    if isinstance(c,str):
        c=openslide.open_slide(c)
    
    if level is None:
        level = c.level_count - 1
    elif level > c.level_count - 1:
        print " level ask is too low... It was setted accordingly"
    sample = c.read_region((0,0), level, c.level_dimensions[level])
    
    return sample
    
    
def get_X_Y(slide,x_0,y_0,level):
    ## Gives you the coordinates for the level 0 image for a given couple of pixel

    size_x_0=slide.level_dimensions[level][0]
    size_y_0=slide.level_dimensions[level][1]
    size_x_1=float(slide.level_dimensions[0][0])
    size_y_1=float(slide.level_dimensions[0][1])
  
    x_1=x_0*size_x_1/size_x_0
    y_1=y_0*size_y_1/size_y_0
    
    return int(x_1),int(y_1)

def get_X_Y_from_0(slide,x_1,y_1,level):
    ## Gives you the coordinates for the level 'level' image for a given couple of pixel from resolution 0

    size_x_0=slide.level_dimensions[level][0]
    size_y_0=slide.level_dimensions[level][1]
    size_x_1=float(slide.level_dimensions[0][0])
    size_y_1=float(slide.level_dimensions[0][1])
  
    x_0=x_1*size_x_0/size_x_1
    y_0=y_1*size_y_0/size_y_1
  

    return int(x_0),int(y_0)
                
                
def get_size(slide,size_x,size_y,level_from,level_to):
    ## Gives comparable size from one level to an other    
    
    ds = slide.level_downsamples
    scal=float(ds[level_from])/ds[level_to]
    size_x_new = int(float(size_x)*scal)
    size_y_new = int(float(size_y)*scal)
    
    return(size_x_new,size_y_new)



def find_square(slide,x_i,y_i,level_resolution,nber_pixels,current_level):
    #for a given pixel, returns a square centered on this pixel of a certain h and w
    ### I could add a white filter, so that shit images stay small

    #pdb.set_trace()
    x_0, y_0 = get_X_Y(slide, x_i ,y_i, current_level)
    if isinstance(nber_pixels, int):
        h = np.ceil(np.sqrt(nber_pixels))
        w = h
    elif isinstance(nber_pixels, tuple):
        h = nber_pixels[0]
        w = nber_pixels[1]
    else:
        raise NameError("Issue number 0002")
        return [0,0,0,0,0]
    w_0, h_0 = get_size(slide,w,h,level_resolution, 0)
    new_x = max(x_0-w_0/2, 0)
    new_y = max(y_0-h_0/2, 0)
    return [int(new_x),int(new_y),int(w),int(h),int(level_resolution)]
