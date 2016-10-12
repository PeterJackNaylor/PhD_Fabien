

from gi.repository import Vips
import openslide
from UsefulOpenSlide import GetImage
from ShortPrediction import Preprocessing, OutputNet


def ApplyToSlideWrite(slide, table, f, outputfilename=None):
                # Slide is a string of the location of the file

        #  This function applies a function f to the whole slide, this slide is given as input with a table
        # which contains all the patches on which to apply the function.
        # Their is also a optionnal outputfilename

        #  table is a iterable where each element has 5 attributes:
        #   x, y, w, h, res

    input_slide = openslide.open_slide(slide)
    outputfilename = outputfilename if outputfilename is not None else "F_" + slide
    dim1, dim2 = input_slide.dimensions
    output_slide = Vips.Image.black(dim1, dim2)

    for i in range(table):
        image = GetImage(input_slide, table[i])
        image = f(image)
        output_slide = output_slide.insert(image, table[i][0], table[i][1])

    #  writing stage
    output_slide.write_to_file(outputfilename)


def GetNet(cn, wd):

    root_directory = wd + "/" + cn + "/"
    if 'FCN' not in el:
        folder = root_directory + "temp_files/"
        weight = folder + "weights." + cn + ".caffemodel"
        deploy = root_directory + "test.prototxt"
    else:
        folder = root_directory + "FCN8/temp_files/"
        weight = folder + "FCN8/weights." + cn + ".caffemodel"
        deploy = root_directory + "FCN8/test.prototxt"

        net = caffe.Net(deploy, weight, caffe.TRAIN)
    return net


def PredImageFromNet(net, image, with_depross=True):
	if with_depross:
		image = Preprocessing(image)
    net.blobs['data'].data[0] = image
    conv1_name = [el for el in net.blobs.keys() if "conv" in el][0]
    new_score = net.forward(["data"],start=conv1_name, end='score')
    bin_map = OutputNet(new_score["score"])
    prob_map = OutputNet(new_score["score"], method="softmax")
    return prob_map, bin_map


if __name__ == '__main__':
    print "In this script, we will take one slide and create a new slide, this new slide will be annotated with cells"

    from TissueSegmentation import ROI_binary_mask
    from ShortPrediction import 
    slide_name = "/data/users/pnaylor/Test_002.tif"
    out_slide = "/data/users/pnaylor/Test_002_pred.tif"

    size_images = 224
    list_of_para = ROI(slide_name, method="grid_fixed_size",
                       ref_level=0, seed=42, fixed_size_in = (size_images, size_images))
    i = 0
    import caffe
    caffe.set_mode_cpu()

    cn_1 = "FCN_randomcrop"
    cn_2 = "batchLAYER4"

    wd_1 = "/data/users/pnaylor/Documents/Python/SoftmaxWithWeight/"
    wd_2 = wd_1

    net_1 = GetNet(cn_1, wd_1)
    net_2 = GetNet(cn_2, wd_2)


    def predict_ensemble(image):
    	prob_image1, bin_image1 = PredImageFromNet(net_1, image, with_depross = True)
		prob_image2, bin_image2 = PredImageFromNet(net_2, image, with_depross = True)
		prob_ensemble = (prob_image1 + prob_image2) / 2
		bin_ensemble = (prob_ensemble > 0.5) + 0
		segmentation_mask = ProbDeprocessing(prob_image, bin_image, param, method="ws_recons")
		contours = dilation(segmentation_mask, square(2)) - erosion(segmentation_mask, square(2))

		x, y = np.where(contours == 1)
		image[x, y, 0] = 255
		image[x, y, 1] = 255
		image[x, y, 2] = 255
		return image 

    	
	start_time = time.time()
	ApplyToSlideWrite(slide_name, list_of_para, predict_ensemble, outputfilename=out_slide)
	diff_time = time.time() - start_time
	print ' \n '
    print 'Time for slide:'
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)

    print ' \n '
    print "Average time per image:"
    diff_time = diff_time / len(list_of_para)
    print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
