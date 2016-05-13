# -*- coding: utf-8 -*-
"""
Command line example : 
%run CuttingPatches.py --file D:/dataThomas/Projet_FR-TNBC-2015-09-30/All\1572_HES_(Colo)_20150925_162438.tiff --output_folder D:/dataThomas/Projet_FR-TNBC-2015-09-30/SlideSegmentation --res 0 --height 512 --width 512 --nber_squares 400 --perc_contour 0.15

"""

import pdb

import time
import os
import random
import openslide
import numpy as np
from scipy import ndimage
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import UsefulOpenSlide as UOS
from TissueSegmentation import ROI_binary_mask, save

from optparse import OptionParser



def Sample_imagette(im_bin, N, slide, level_resolution, nber_pixels, current_level,mask):
    ### I should modify this function, so that the squares don't fall on each other.. 
    y, x = np.where(im_bin>0)
    n = len(x)
    indices = range(n)
    random.shuffle(indices)
    #indices=indices[0:N]
    result = []
    i = 0
    #pdb.set_trace()
    while i < n and len(result) < N:
        x_i = x[indices[i]]
        y_i = y[indices[i]]
        if mask[y_i,x_i ]== 0:
            para = UOS.find_square(slide,x_i,y_i,level_resolution,nber_pixels,current_level)
            result.append(para)
            x_,y_ = UOS.get_X_Y_from_0(slide,para[0],para[1],current_level)
            w_,h_ = UOS.get_size(slide,para[2],para[3],level_resolution,current_level)
            add = int(w_/2)      ### allowing how much overlapping?
            mask[max( (y_-add), 0):min( (y_+add+h_), im_bin.shape[0] ), max((x_-add),0):min( (x_+add+w_), im_bin.shape[1] )] = 1
        i +=1            
    return(result,mask)

def White_score(slide,para,thresh):
    crop=UOS.GetImage(slide,para) 
    #pdb.set_trace()
    crop=np.array(crop)[:,:,0]
    binary = crop > thresh
    nber_ones=sum(sum(binary))
    nber_total=binary.shape[0]*binary.shape[1]
    return(float(nber_ones)/nber_total)
    
def Best_Finder_rec(slide,level,x_0,y_0,size_x,size_y,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge=0):
    ## This function enables you to cut up a portion of a slide to a given resolution. 
    ## It will try and minimize the number of create images for a given portion of the image
    if size_x*size_y==0:
        print 'Warning: width or height is null..'
        return([])
    else:
        if level==ref_level:          
            if size_x*size_y<number_of_pixels_max: ##size of level 3
                if marge>0:
                    if isinstance(marge,int):  ## if it is int, then it ill add that amount of pixels
                        extra_pixels=marge
                    elif isinstance(marge,float): ## if it is float it will multiply it with respect to its size. 
                        extra_pixels=int(np.ceil(marge*min(size_x,size_y)))
                    size_x +=extra_pixels/2
                    size_y +=extra_pixels/2
                    width_xp,height_xp=UOS.get_size(slide,extra_pixels/2,extra_pixels/2,level,0)
                    x_0 = max(x_0 - width_xp,0)
                    y_0 = max(y_0 - height_xp,0)
                para=[x_0,y_0,size_x,size_y,level]
                if White_score(slide,para,thresh)<0.5:
                    list_roi.append(para)
                return(list_roi)
            else:
                            
                size_x_new=int(size_x*0.5)
                size_y_new=int(size_y*0.5)
                    
                diese_str="#"*level*10
                if verbose:
                    print diese_str +"split level "+ str(level)
                
                width_x_0,height_y_0=UOS.get_size(slide,size_x,size_y,level,0)                
                x_1=x_0+int(width_x_0*0.5)
                y_1=y_0+int(height_y_0*0.5)
                
                image_name=image_name+"_Split_id_"+str(random.randint(0, 1000))            
                list_roi=Best_Finder_rec(slide,level,x_0,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                list_roi=Best_Finder_rec(slide,level,x_1,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                list_roi=Best_Finder_rec(slide,level,x_0,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                list_roi=Best_Finder_rec(slide,level,x_1,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                return(list_roi)
        else:
            if level > 1:
                if size_x*size_y>number_of_pixels_max: ##size of level 3
                    
                    size_x_new,size_y_new=UOS.get_size(slide,size_x,size_y,level,level-1)
                    size_x_new=int(size_x_new*0.5)
                    size_y_new=int(size_y_new*0.5)
                    
                    if verbose:
                        diese_str="#"*level*10
                        print diese_str +"split level "+ str(level)
                    
                    width_x_0,height_y_0=UOS.get_size(slide,size_x,size_y,level,0)
                    
                    x_1=x_0+int(width_x_0*0.5)
                    y_1=y_0+int(height_y_0*0.5)
                    
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    list_roi=Best_Finder_rec(slide,level-1,x_1,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    list_roi=Best_Finder_rec(slide,level-1,x_1,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    return(list_roi)
                else:
                    
                    size_x_new,size_y_new=UOS.get_size(slide,size_x,size_y,level,level-1)
                    
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    return(list_roi)
            else:
                print "Not enough variability on second split"
                

def ROI(name,ref_level=4, disk_size=4, thresh=None, black_spots=None,
        number_of_pixels_max=1000000, verbose=False, marge=0, method='grid',
        mask_address=None, contour_size=3, N_squares=100, seed=None):   
    ## creates a grid of the all interesting places on the image

    if seed is not None:
        random.seed(seed)

    if '/' in name:
        cut=name.split('/')[-1]
        folder=cut.split('.')[0]
    else:
        folder=name.split(".")[0] 
    slide = openslide.open_slide(name)
    list_roi=[]
    #pdb.set_trace()

    if method=='grid':
        lowest_res=len(slide.level_dimensions)-2
    
        s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,0:3]
        
        binary=ROI_binary_mask(s)
        stru = [[1,1,1],[1,1,1],[1,1,1]]
        blobs, number_of_blobs = ndimage.label(binary,structure=stru)
        for i in range(1,number_of_blobs+1):
            y,x=np.where(blobs == i)
            x_0=min(x)
            y_0=min(y)
            w=max(x)-x_0
            h=max(y)-y_0               
            new_x,new_y=UOS.get_X_Y(slide,x_0,y_0,lowest_res)
            list_roi=Best_Finder_rec(slide,lowest_res,new_x,new_y,w,h,"./"+folder+"/"+folder,ref_level,list_roi,number_of_pixels_max,thresh,verbose) 
            

    elif method=='grid_etienne':
        lowest_res=len(slide.level_dimensions)-2
    
        s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,0:3]
        
        binary=ROI_binary_mask(s)
        stru = [[1,1,1],[1,1,1],[1,1,1]]
        blobs, number_of_blobs = ndimage.label(binary,structure=stru)
        for i in range(1,number_of_blobs+1):
            y,x=np.where(blobs == i)
            x_0=min(x)
            y_0=min(y)
            w=max(x)-x_0
            h=max(y)-y_0               
            new_x,new_y=UOS.get_X_Y(slide,x_0,y_0,lowest_res)
            list_roi=Best_Finder_rec(slide,lowest_res,new_x,new_y,w,h,"./"+folder+"/"+folder,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
            
            
    elif method=='SP_ROI':        
        
        lowest_res=len(slide.level_dimensions)-2

        s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,0:3]
        
        binary = ROI_binary_mask(s)
        binary[binary>0] = 255
        
        uniq, counts = np.unique(binary, return_counts=True)
        background_val = uniq[np.argmax(counts)]
        binary[binary!=background_val] = background_val  +1
        binary -= background_val


        contour_binary      = ndimage.morphology.morphological_gradient(binary, size = (contour_size, contour_size) )
        list_roi            = []
        mask                = np.zeros( shape=(binary.shape[0], binary.shape[1]), dtype='uint8' )
        
        if isinstance(N_squares, int):
            n_1 = N_squares / 2
            n_2 = n_1
        elif isinstance(N_squares, tuple):
            n_1 = N_squares[0]
            n_2 = N_squares[1]
        else:
            raise NameError("Issue number 0001")
            return []
            
        list_outside, mask        = Sample_imagette(binary, n_1, slide, ref_level,
                                                    number_of_pixels_max, lowest_res, mask)
        list_contour_binary, mask = Sample_imagette(contour_binary, n_2, slide, ref_level, 
                                                    number_of_pixels_max, lowest_res, mask)
        list_roi                  = list_outside + list_contour_binary
    else:
        raise NameError("Not known method")


    list_roi=np.array(list_roi)
    return(list_roi)

def visualise_cut(slide,list_pos,res_to_view=None,color='red',size=12,title=""):
    if res_to_view is None:
        res_to_view=slide.level_count-3
    whole_slide=np.array(slide.read_region((0,0),res_to_view,slide.level_dimensions[res_to_view]))
    max_x,max_y=slide.level_dimensions[res_to_view]
    fig = plt.figure(figsize=(size,size ))
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(whole_slide)#,origin='lower')
    for para in list_pos:
        top_left_x,top_left_y=UOS.get_X_Y_from_0(slide,para[0],para[1],res_to_view)
        w,h=UOS.get_size(slide,para[2],para[3],para[4],res_to_view)
        p=patches.Rectangle((top_left_x,max_y-top_left_y-h), w, h, fill=False, edgecolor=color)
        p=patches.Rectangle((top_left_x,top_left_y), w, h, fill=False, edgecolor=color)
        ax.add_patch(p)
    ax.set_title(title, size=20)
    plt.show()

if __name__ == "__main__":
    
    parser = OptionParser()

    parser.add_option("-f", "--file", dest="file",
                      help="Input file (raw data)")
    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="Where to store the patches")
    parser.add_option('-r', '--res', dest="resolution",
                      help="Resolution")
    parser.add_option('--height', dest='h',
                      help="height of an image")
    parser.add_option('--width', dest='w',
                      help="width of an image")
    parser.add_option('--nber_squares', dest='N',
                      help="Number of squares")
    parser.add_option('--perc_contour', dest="perc",
                      help="percentage of squares given to the contours", default = "0.2")
    (options, args) = parser.parse_args()
    
    ## checking the input data
    try:
        slide = openslide.open_slide(options.file)
    except:
        print "Issue with file name"
    try:
        if not os.path.isdir(options.output_folder):
            os.mkdir(options.output_folder)
        name = options.file.split('\\')[-1].split('.')[0]
        options.output_folder = options.output_folder +"\\"+ name
        if not os.path.isdir(options.output_folder):
            os.mkdir(options.output_folder)
    except:
        print "Failed to check output folder..."
    try:
        options.resolution = int(options.resolution)
        options.h = int(options.h)
        options.w = int(options.w)
        options.N = int(options.N)
    except:
        print "Problem while converting to int.." 
    try:
        options.perc = float(options.perc)
        n_1 = int(options.N * ( 1 - options.perc))
        n_2 = options.N - n_1
    except:
        print "Problem while converting to float.."
    
    print "Input paramters to CuttingPatches:"
    print " \n "
    print "Input file        : | " + options.file
    print "Output folder     : | " + options.output_folder
    print "Resolution        : | " + str(options.resolution)
    print "Height of img     : | " + str(options.h)
    print "Width of img      : | " + str(options.w)
    print "Nber on the inside: | " + str(n_1)
    print "Nber on the contour:| " + str(n_2)
    
    print ' \n '       
    print "Beginning analyse:" 
    
    start_time = time.time()
### Core of the code    
########################################
    
    list_of_para = ROI(options.file, method = "SP_ROI", ref_level = options.resolution,
                       N_squares = (n_1, n_2), seed = 42, number_of_pixels_max = (options.w, options.h))
    i = 0
    for para in list_of_para:
        sample = UOS.GetImage(options.file, para)
        sample.save(options.output_folder + "\\" + str(i)+".png")
        i += 1
########################################
    
    diff_time = time.time() - start_time

    print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
    
    print ' \n '
    print "Average time per image:" 
    diff_time = diff_time / len(list_of_para)
    print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)