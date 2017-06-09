from ObjectOriented import ConvolutionalNeuralNetwork
import tensorflow as tf
import os
import numpy as np
from UsefulFunctions.ImageTransf import ListTransform
from DataGen2 import DataGen
import pdb

def print_dim(text ,tensor):
    print text, tensor.get_shape()
    print
class UNet(ConvolutionalNeuralNetwork):

    def conv_layer_f(self, i_layer, w_var, scope_name="conv", strides=[1,1,1,1], padding="VALID"):
        with tf.name_scope(scope_name):
            return tf.nn.conv2d(i_layer, w_var, strides=strides, padding=padding)


    def transposeconv_layer_f(self, i_layer, w_, scope_name="tconv", padding="VALID"):
        i_shape = tf.shape(i_layer)
        o_shape = tf.stack([i_shape[0], i_shape[1]*2, i_shape[2]*2, i_shape[3]//2])
        return tf.nn.conv2d_transpose(i_layer, w_, output_shape=o_shape,
                                            strides=[1,2,2,1], padding=padding)

    def max_pool(self, i_layer, ksize=[1,2,2,1], strides=[1,2,2,1],
                 padding="VALID", name="MaxPool"):
        return tf.nn.max_pool(i_layer, ksize=ksize, strides=strides, 
                              padding=padding, name=name)

    def CropAndMerge(self, Input1, Input2, name="bridge"):
        #pdb.set_trace()
        Size1 = tf.shape(Input1)[1]
        Size2 = tf.shape(Input2)[1]
        ### Size1 > Size2
        with tf.name_scope(name):
            #assert Size1 > Size2 
            diff = tf.divide(tf.subtract(Size1, Size2), 2)
            diff = tf.cast(diff, tf.int32)
            Size2 = tf.cast(Size2, tf.int32)
            crop = tf.slice(Input1, [0, diff, diff, 0], [-1, Size2, Size2, -1])
            concat = tf.concat([crop, Input2], axis=3)

            return concat
    def input_node_f(self):
        return tf.placeholder(
               tf.float32,
               shape=(self.BATCH_SIZE, self.IMAGE_SIZE + 184, self.IMAGE_SIZE + 184, self.NUM_CHANNELS))


    def init_vars(self):
        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()
        n_features = 64

        self.conv1_1weights = self.weight_xavier(3, self.NUM_CHANNELS, n_features, "conv1_1/")
        self.conv1_1biases = self.biases_const_f(0.1, n_features, "conv1_1/")

        self.conv1_2weights = self.weight_xavier(3, n_features, n_features, "conv1_2/")
        self.conv1_2biases = self.biases_const_f(0.1, n_features, "conv1_2/")

        self.conv1_3weights = self.weight_xavier(3, 2 * n_features, n_features, "conv1_3/")
        self.conv1_3biases = self.biases_const_f(0.1, n_features, "conv1_3/")

        self.conv1_4weights = self.weight_xavier(3, n_features, n_features, "conv1_4/")
        self.conv1_4biases = self.biases_const_f(0.1, n_features, "conv1_4/")



        self.conv2_1weights = self.weight_xavier(3, n_features, 2 * n_features, "conv2_1/")
        self.conv2_1biases = self.biases_const_f(0.1, 2 * n_features, "conv2_1/")

        self.conv2_2weights = self.weight_xavier(3, 2 * n_features, 2 * n_features, "conv2_2/")
        self.conv2_2biases = self.biases_const_f(0.1, 2 * n_features, "conv2_2/")

        self.conv2_3weights = self.weight_xavier(3, 4 * n_features, 2 * n_features, "conv2_3/")
        self.conv2_3biases = self.biases_const_f(0.1, 2 * n_features, "conv2_3/")

        self.conv2_4weights = self.weight_xavier(3, 2 * n_features, 2 * n_features, "conv2_4/")
        self.conv2_4biases = self.biases_const_f(0.1, 2 * n_features, "conv2_4/")



        self.conv3_1weights = self.weight_xavier(3, 2 * n_features, 4 * n_features, "conv3_1/")
        self.conv3_1biases = self.biases_const_f(0.1, 4 * n_features, "conv3_1/")

        self.conv3_2weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_2/")
        self.conv3_2biases = self.biases_const_f(0.1, 4 * n_features, "conv3_2/")

        self.conv3_3weights = self.weight_xavier(3, 8 * n_features, 4 * n_features, "conv3_3/")
        self.conv3_3biases = self.biases_const_f(0.1, 4 * n_features, "conv3_3/")

        self.conv3_4weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_4/")
        self.conv3_4biases = self.biases_const_f(0.1, 4 * n_features, "conv3_4/")



        self.conv4_1weights = self.weight_xavier(3, 4 * n_features, 8 * n_features, "conv4_1/")
        self.conv4_1biases = self.biases_const_f(0.1, 8 * n_features, "conv4_1/")

        self.conv4_2weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_2/")
        self.conv4_2biases = self.biases_const_f(0.1, 8 * n_features, "conv4_2/")

        self.conv4_3weights = self.weight_xavier(3, 16 * n_features, 8 * n_features, "conv4_3/")
        self.conv4_3biases = self.biases_const_f(0.1, 8 * n_features, "conv4_3/")

        self.conv4_4weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_4/")
        self.conv4_4biases = self.biases_const_f(0.1, 8 * n_features, "conv4_4/")



        self.conv5_1weights = self.weight_xavier(3, 8 * n_features, 16 * n_features, "conv5_1/")
        self.conv5_1biases = self.biases_const_f(0.1, 16 * n_features, "conv5_1/")

        self.conv5_2weights = self.weight_xavier(3, 16 * n_features, 16 * n_features, "conv5_2/")
        self.conv5_2biases = self.biases_const_f(0.1, 16 * n_features, "conv5_2/")




        self.tconv5_4weights = self.weight_xavier(2, 8 * n_features, 16 * n_features, "tconv5_4/")
        self.tconv5_4biases = self.biases_const_f(0.1, 8 * n_features, "tconv5_4/")

        self.tconv4_3weights = self.weight_xavier(2, 4 * n_features, 8 * n_features, "tconv4_3/")
        self.tconv4_3biases = self.biases_const_f(0.1, 4 * n_features, "tconv4_3/")

        self.tconv3_2weights = self.weight_xavier(2, 2 * n_features, 4 * n_features, "tconv3_2/")
        self.tconv3_2biases = self.biases_const_f(0.1, 2 * n_features, "tconv3_2/")

        self.tconv2_1weights = self.weight_xavier(2, n_features, 2 * n_features, "tconv2_1/")
        self.tconv2_1biases = self.biases_const_f(0.1, n_features, "tconv2_1/")



        self.logits_weight = self.weight_xavier(1, n_features, 2, "logits/")
        self.logits_biases = self.biases_const_f(0.1, 2, "logits/")

        self.keep_prob = tf.Variable(0.5, name="dropout_prob")

        print('Model variables initialised')



    def init_model_architecture(self):

        self.conv1_1 = self.conv_layer_f(self.input_node, self.conv1_1weights, "conv1_1/")
        self.relu1_1 = self.relu_layer_f(self.conv1_1, self.conv1_1biases, "conv1_1/")

        self.conv1_2 = self.conv_layer_f(self.relu1_1, self.conv1_2weights, "conv1_2/")
        self.relu1_2 = self.relu_layer_f(self.conv1_2, self.conv1_2biases, "conv1_2/")


        self.pool1_2 = self.max_pool(self.relu1_2, name="pool1_2")


        self.conv2_1 = self.conv_layer_f(self.pool1_2, self.conv2_1weights, "conv2_1/")
        self.relu2_1 = self.relu_layer_f(self.conv2_1, self.conv2_1biases, "conv2_1/")

        self.conv2_2 = self.conv_layer_f(self.relu2_1, self.conv2_2weights, "conv2_2/")
        self.relu2_2 = self.relu_layer_f(self.conv2_2, self.conv2_2biases, "conv2_2/")        


        self.pool2_3 = self.max_pool(self.relu2_2, name="pool2_3")


        self.conv3_1 = self.conv_layer_f(self.pool2_3, self.conv3_1weights, "conv3_1/")
        self.relu3_1 = self.relu_layer_f(self.conv3_1, self.conv3_1biases, "conv3_1/")

        self.conv3_2 = self.conv_layer_f(self.relu3_1, self.conv3_2weights, "conv3_2/")
        self.relu3_2 = self.relu_layer_f(self.conv3_2, self.conv3_2biases, "conv3_2/")     


        self.pool3_4 = self.max_pool(self.relu3_2, name="pool3_4")


        self.conv4_1 = self.conv_layer_f(self.pool3_4, self.conv4_1weights, "conv4_1/")
        self.relu4_1 = self.relu_layer_f(self.conv4_1, self.conv4_1biases, "conv4_1/")

        self.conv4_2 = self.conv_layer_f(self.relu4_1, self.conv4_2weights, "conv4_2/")
        self.relu4_2 = self.relu_layer_f(self.conv4_2, self.conv4_2biases, "conv4_2/")


        self.pool4_5 = self.max_pool(self.relu4_2, name="pool4_5")


        self.conv5_1 = self.conv_layer_f(self.pool4_5, self.conv5_1weights, "conv5_1/")
        self.relu5_1 = self.relu_layer_f(self.conv5_1, self.conv5_1biases, "conv5_1/")

        self.conv5_2 = self.conv_layer_f(self.relu5_1, self.conv5_2weights, "conv5_2/")
        self.relu5_2 = self.relu_layer_f(self.conv5_2, self.conv5_2biases, "conv5_2/")



        self.tconv5_4 = self.transposeconv_layer_f(self.relu5_2, self.tconv5_4weights, "tconv5_4/")
        self.trelu5_4 = self.relu_layer_f(self.tconv5_4, self.tconv5_4biases, "tconv5_4/")
        self.bridge4 = self.CropAndMerge(self.relu4_2, self.trelu5_4, "bridge4")



        self.conv4_3 = self.conv_layer_f(self.bridge4, self.conv4_3weights, "conv4_3/")
        self.relu4_3 = self.relu_layer_f(self.conv4_3, self.conv4_3biases, "conv4_3/")

        self.conv4_4 = self.conv_layer_f(self.relu4_3, self.conv4_4weights, "conv4_4/")
        self.relu4_4 = self.relu_layer_f(self.conv4_4, self.conv4_4biases, "conv4_4/")



        self.tconv4_3 = self.transposeconv_layer_f(self.relu4_4, self.tconv4_3weights, "tconv4_3/")
        self.trelu4_3 = self.relu_layer_f(self.tconv4_3, self.tconv4_3biases, "tconv4_3/")
        self.bridge3 = self.CropAndMerge(self.relu3_2, self.trelu4_3, "bridge3")



        self.conv3_3 = self.conv_layer_f(self.bridge3, self.conv3_3weights, "conv3_3/")
        self.relu3_3 = self.relu_layer_f(self.conv3_3, self.conv3_3biases, "conv3_3/")

        self.conv3_4 = self.conv_layer_f(self.relu3_3, self.conv3_4weights, "conv3_4/")
        self.relu3_4 = self.relu_layer_f(self.conv3_4, self.conv3_4biases, "conv3_4/")



        self.tconv3_2 = self.transposeconv_layer_f(self.relu3_4, self.tconv3_2weights, "tconv3_2/")
        self.trelu3_2 = self.relu_layer_f(self.tconv3_2, self.tconv3_2biases, "tconv3_2/")
        self.bridge2 = self.CropAndMerge(self.relu2_2, self.trelu3_2, "bridge2")



        self.conv2_3 = self.conv_layer_f(self.bridge2, self.conv2_3weights, "conv2_3/")
        self.relu2_3 = self.relu_layer_f(self.conv2_3, self.conv2_3biases, "conv2_3/")

        self.conv2_4 = self.conv_layer_f(self.relu2_3, self.conv2_4weights, "conv2_4/")
        self.relu2_4 = self.relu_layer_f(self.conv2_4, self.conv2_4biases, "conv2_4/")



        self.tconv2_1 = self.transposeconv_layer_f(self.relu2_4, self.tconv2_1weights, "tconv2_1/")
        self.trelu2_1 = self.relu_layer_f(self.tconv2_1, self.tconv2_1biases, "tconv2_1/")
        self.bridge1 = self.CropAndMerge(self.relu1_2, self.trelu2_1, "bridge1")



        self.conv1_3 = self.conv_layer_f(self.bridge1, self.conv1_3weights, "conv1_3/")
        self.relu1_3 = self.relu_layer_f(self.conv1_3, self.conv1_3biases, "conv1_3/")

        self.conv1_4 = self.conv_layer_f(self.relu1_3, self.conv1_4weights, "conv1_4/")
        self.relu1_4 = self.relu_layer_f(self.conv1_4, self.conv1_4biases, "conv1_4/")
        self.last = self.relu1_4

        print('Model architecture initialised')



if __name__== "__main__":

    CUDA_NODE = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)

    SAVE_DIR = "/tmp/object/unet/long_0.001"
    N_ITER_MAX = 2000
    N_TRAIN_SAVE = 100
    N_TEST_SAVE = 100
    LEARNING_RATE = 0.001
    MEAN = np.array([122.67892, 116.66877 ,104.00699])
    HEIGHT = 212 
    WIDTH = 212
    CROP = 4
    PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
    PATH = '/home/pnaylor/Documents/Data/ToAnnotate'
    PATH = "/data/users/pnaylor/Bureau/ToAnnotate"
#    PATH = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotate"
    BATCH_SIZE = 2
    LRSTEP = "epoch/2"
    SUMMARY = True
    S = SUMMARY


    transform_list, transform_list_test = ListTransform()
    DG_TRAIN = DataGen(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True)
    
    N_ITER_MAX = 200 * DG_TRAIN.length

    DG_TEST  = DataGen(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True)

    model = UNet(LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=HEIGHT,
                                       NUM_LABELS=2,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42)

    model.train(DG_TRAIN, DG_TEST)