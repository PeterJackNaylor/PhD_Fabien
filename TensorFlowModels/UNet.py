import numpy as np
import tensorflow as tf
from Data.DataGen import DataGen
from UsefulFunctions import ImageTransf as Transf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import pdb

LEARNING_RATE = 0.01

transform_list = [Transf.Identity(),
                  Transf.Flip(0),
                  Transf.Flip(1)]

width = 224
height = 224
dim = 3 
batch_size = 2
MEAN = np.array([104.00699, 116.66877, 122.67892])


dg = DataGen('/data/users/pnaylor/Bureau/ToAnnotate', crop = 4, 
             size=(width, height), transforms=transform_list, Unet=True)
dg_test = DataGen('/data/users/pnaylor/Bureau/ToAnnotate', split="test",
                  crop = 4, size=(width, height), transforms=transform_list
                  , Unet=True)


def Conv(input_layer, nout, ks, padding = "same"):
    return tf.layers.conv2d(
        inputs=input_layer,
        filters=nout,
        kernel_size=[ks, ks],
        padding=padding,
        activation=None))

def Maxpool(input_layer, ks = 2, stride=2, padding='VALID'):
    return tf.layers.max_pool2d(input_layer, 
                                kernel_size=[ks,ks],
                                stride =[stride,stride],
                                padding=padding)

def ConvBnRelu(input_layer, nout, ks, padding = "same"):
    conv = Conv(input_layer, nout, ks, padding)
    bn = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(bn)
    return conv, bn, relu

def DeconvBnRelu(input_layer, nout, ks, padding = "VALID"):
    deconv = tf.contrib.layers.conv2d_transpose(input_layer,
                                                nout,
                                                kernel_size = [ks, ks],
                                                padding)
    bn = tf.layers.batch_normalization(deconv)
    relu = tf.nn.relu(bn)
    return deconv, bn, relu


layers.conv2d_transpose

def Crop(input_layer, size):
    return tf.image.resize_image_with_crop_or_pad(input_layer, size, size)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, width, height, dim])

    # Convolutional Layer #1
    conv1_1, bn1_1, relu1_1 = ConvBnRelu(input_layer, 64, 3, "VALID") # n - 4
    conv1_2, bn1_2, relu1_2 = ConvBnRelu(relu1_1, 64, 3, "VALID") # n - 8
    n1 = width - 8
    pool1 = Maxpool(relu1_2) # (n - 8) /2

    conv2_1, bn2_1, relu2_1 = ConvBnRelu(pool1, 128, 3, "VALID") # ((n-8)/2) - 4
    conv2_2, bn2_2, relu2_2 = ConvBnRelu(relu2_1, 128, 3, "VALID") # ((n-8)/2) - 8
    n2 = n1 / 2 - 8
    pool2 = Maxpool(relu2_2) # (((n-8)/2) - 8) /2

    conv3_1, bn3_1, relu3_1 = ConvBnRelu(pool2, 256, 3, "VALID") # (((n-8)/2) - 8) /2 - 4
    conv3_2, bn3_2, relu3_2 = ConvBnRelu(relu3_1, 256, 3, "VALID") # (((n-8)/2) - 8) /2 - 8
    n3 = n2 / 2 - 8
    pool3 = Maxpool(relu3_2) # ((((n-8)/2) - 8) /2 - 8 ) / 2


    conv4_1, bn4_1, relu4_1 = ConvBnRelu(pool3, 512, 3, "VALID") # ((((n-8)/2)-8)/2-8)/2 - 4
    conv4_2, bn4_2, relu4_2 = ConvBnRelu(relu4_1, 512, 3, "VALID") # ((((n-8)/2)-8)/2-8)/2 - 8
    n4 = n3 / 2 - 8
    pool4 = Maxpool(relu4_2) #(((((n-8)/2)-8)/2-8)/2 - 8)/2

    conv5_1, bn5_1, relu5_1 = ConvBnRelu(pool4, 1024, 3, "VALID") #(((((n-8)/2)-8)/2-8)/2 - 8)/2 - 4
    conv5_2, bn5_2, relu5_2 = ConvBnRelu(relu5_1, 1024, 3, "VALID") #(((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8
    n5 = n4 / 2 - 8
    deconv_To_4, bn_To_4, relu_To_4 = DeconvBnRelu(relu5_2, 512, 2, padding = "VALID")
                                ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2

    n4_deconv = n5 * 2 
    diff4 = (n4 - n4_deconv) / 2
    
    crop_relu4_2 = tf.slice(relu4_2, [0, diff4, diff4, -1], [-1, n4_deconv, n4_deconv, -1])
    concat4 = tf.concat([relu_To_4, crop_relu4_2])

    conv4_3, bn4_3, relu4_3 = ConvBnRelu(concat4, 512, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv4_4, bn4_4, relu4_4 = ConvBnRelu(relu4_3, 512, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_3, bn_To_3, relu_To_3 = DeconvBnRelu(relu4_4, 256, 2, padding = "VALID")
    
    n3_deconv = (n4_deconv - 8) * 2
    diff3 = (n3 - n3_deconv) / 2

    crop_relu3_2 = tf.slice(relu3_2, [0, diff3, diff3, -1], [-1, n3_deconv, n3_deconv, -1])
    concat3 = tf.concat([relu_To_3, crop_relu3_2])

    conv3_3, bn3_3, relu3_3 = ConvBnRelu(concat3, 256, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv3_4, bn3_4, relu3_4 = ConvBnRelu(relu3_3, 256, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_2, bn_To_2, relu_To_2 = DeconvBnRelu(relu4_4, 128, 2, padding = "VALID")

    n2_deconv = (n3_deconv - 8) * 2
    diff2 = (n2 - n2_deconv) / 2 

    crop_relu2_2 = tf.slice(relu2_2, [0, diff2, diff2, -1], [-1, n2_deconv, n2_deconv, -1])
    concat2 = tf.concat([relu_To_2, crop_relu2_2]) 

    conv2_3, bn2_3, relu2_3 = ConvBnRelu(concat2, 128, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv2_4, bn2_4, relu2_4 = ConvBnRelu(relu2_3, 128, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    deconv_To_1, bn_To_1, relu_To_1 = DeconvBnRelu(relu2_4, 64, 2, padding = "VALID")

    n1_deconv = (n2_deconv - 8) * 2
    diff2 = (n1 - n1_deconv) / 2 

    crop_relu1_2 = tf.slice(relu1_2, [0, diff1, diff1, -1], [-1, n1_deconv, n1_deconv, -1])
    concat1 = tf.concat([relu_To_1, crop_relu1_2]) 

    conv1_3, bn1_3, relu1_3 = ConvBnRelu(concat1, 64, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 4
    conv1_4, bn1_4, relu1_4 = ConvBnRelu(relu1_3, 64, 3, "VALID") ##((((((n-8)/2)-8)/2-8)/2 - 8)/2 - 8) * 2 - 8
    
    logit = Conv(relu1_4, 2, 1, padding = "same")

    train_op = None
    loss = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

      # Generate Predictions
    predictions = {
          "classes": tf.argmax(
              input=logits, axis=3),
          "probabilities": tf.nn.softmax(
              logits, name="softmax_tensor"),
          "input": input_layer
    }

      # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
          mode=mode, predictions=predictions, loss=loss, train_op=train_op)