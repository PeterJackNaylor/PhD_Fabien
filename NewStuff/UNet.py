# -*- coding: utf-8 -*-

import tensorflow as tf
import BasicNetTF as PTF
import numpy as np
from DataGen2 import DataGen, ListTransform







SAVE_DIR = "/share/data40T_v2/Peter/tmp_tensorflow/retry"
N_ITER_MAX = 1000
N_RECORD = 100
LEARNING_RATE = 0.01
MEAN = np.array([104.00699, 116.66877, 122.67892])
HEIGHT = 212 
WIDTH = 212
CROP = 4
PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
BATCH_SIZE = 4

transform_list, transform_list_test = ListTransform()
DG_TRAIN = DataGen(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                   transforms=transform_list, UNet=True)
DG_TEST  = DataGen(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                   transforms=transform_list_test, UNet=True)


def FeedDict(Train, Inputs, Labels, DGTrain=DG_TRAIN, DGTest=DG_TEST, 
              BatchSize=BATCH_SIZE, Width=WIDTH, Height=HEIGHT,
              Mean=MEAN, Dim=3):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

    ImagesBatch = np.zeros(shape=(BatchSize, Width, Height, Dim))
    LabelsBatch = np.zeros(shape=(BatchSize, Width - 184, Height - 184))   

    if Train:

        for i in range(BatchSize):

            key = DGTrain.NextKeyRandList(0)
            ImagesBatch[i], LabelsBatch[i] = DGTrain[key]
            ImagesBatch[i] -= Mean

    else:

        for i in range(BatchSize):

            key = DGTest.NextKeyRandList(0)
            ImagesBatch[i], LabelsBatch[i] = DGTest[key]   
            ImagesBatch[i] -= Mean

    return {Inputs: ImagesBatch, Labels: LabelsBatch}



def Training(LearningRate, ImageSizeIn, ImageSizeOut):

    Logits, Inputs, Labels = UNetModel(ImageSizeIn, ImageSizeOut)
    Predicted = tf.argmax(input=Logits, axis=3)
    Labels = tf.to_int64(Labels)

    with tf.name_scope('Loss'):

        with tf.name_scope('Cross Entropy'):

            CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=Labels, logits=Logits, name='xentropy')

        with tf.name_scope('Loss'):

            Loss = tf.reduce_mean(CrossEntropy, name='xentropy_mean')

        tf.summary.scalar('Loss', Loss)

    with tf.name_scope('Accuracy'):

        with tf.name_scope('Correct prediction'):

            correct_prediction = tf.equal(tf.argmax(Predicted, 1), tf.argmax(Labels, 1))

        with tf.name_scope('Accuracy'):

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('F1'):

            with tf.name_scope('Subs'):

                TP = tf.count_nonzero(Predicted * Labels)
                TN = tf.count_nonzero((Predicted - 1) * (Labels - 1))
                FP = tf.count_nonzero(Predicted * (Labels - 1))
                FN = tf.count_nonzero((Predicted - 1) * Labels)

            with tf.name_scope('Precision'):

                precision = tf.divide(TP, tf.add(TP, FP))

            with tf.name_scope('Recall'):

                recall = tf.divide(TP, tf.add(TP, TN))

            with tf.name_scope('F1'):

                num = tf.multiply(precision, recall)
                dem = tf.add(precision, recall)
                F1 = tf.scalar_mul(2, tf.divide(num, dem))

        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.scalar('Precision', precision)
        tf.summary.scalar('Recall', recall)
        tf.summary.scalar('F1', F1)



    with tf.name_scope('Training'):

        optimizer = tf.train.GradientDescentOptimizer(LearningRate)

    tf.summary.scalar('Learning rate', LearningRate)

    return loss, optimizer, accuracy, F1, recall, precision, Inputs, Labels


def UNetModel(ImageSizeIn, ImageSizeOut):

    ks = 3
    padding = "VALID"

    Input, Label = PTF.InputLayer(ImageSizeIn, ImageSizeOut, 3, 1, Weight=False)
    
    Out1_1 = PTF.ConvLayer(Input, 3, 64, ks, "Conv1_1", padding)
    Out1_2 = PTF.ConvLayer(Out1_1, 64, 64, ks, "Conv1_2", padding)
    ImageSize1_2 = ImageSizeIn - 4
    MaxPool1 = tf.nn.max_pool(Out1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool1")

    Out2_1 = PTF.ConvLayer(MaxPool1, 64, 128, ks, "Conv2_1", padding)
    Out2_2 = PTF.ConvLayer(Out2_1, 128, 128, ks, "Conv2_2", padding)
    ImageSize2_2 = ImageSize1_2 / 2 - 4
    MaxPool2 = tf.nn.max_pool(Out2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool2")

    Out3_1 = PTF.ConvLayer(MaxPool2, 128, 256, ks, "Conv3_1", padding)
    Out3_2 = PTF.ConvLayer(Out3_1, 256, 256, ks, "Conv3_2", padding)
    ImageSize3_2 = ImageSize2_2 / 2 - 4
    MaxPool3 = tf.nn.max_pool(Out3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool3")

    Out4_1 = PTF.ConvLayer(MaxPool3, 256, 512, ks, "Conv4_1", padding)
    Out4_2 = PTF.ConvLayer(Out4_1, 512, 512, ks, "Conv4_2", padding)
    ImageSize4_2 = ImageSize3_2 / 2 - 4
    MaxPool4 = tf.nn.max_pool(Out4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool4")

    Out5_1 = PTF.ConvLayer(MaxPool4, 512, 1024, ks, "Conv5_1", padding)
    Out5_2 = PTF.ConvLayer(Out5_1, 1024, 1024, ks, "Conv5_2", padding)
    ImageSizeTransposeConv4 = (ImageSize4_2 / 2 - 4) * 2
    TranCon4 = PTF.TransposeConvLayer(Out5_2, 1024, 512, ImageSizeTransposeConv4, ks, "TransposeConv4", "VALID")

    #### 
    Concat4 = PTF.CropAndMerge(Out4_2, TranCon4, ImageSize4_2, ImageSizeTransposeConv4, 512, "Concat4") # Concat has 1024 features maps
    Out4_3 = PTF.ConvLayer(Concat4, 1024, 512, ks, "Conv4_3", padding)
    Out4_4 = PTF.ConvLayer(Out4_3, 512, 512, ks, "Conv4_4", padding)
    ImageSizeTransposeConv3 = (ImageSizeTransposeConv4 - 4) * 2
    TranCon3 = PTF.TransposeConvLayer(Out4_4, 512, 256, ImageSizeTransposeConv3, ks, "TransposeConv3", "VALID")

    ####
    Concat3 = PTF.CropAndMerge(Out3_2, TranCon3, ImageSize3_2, ImageSizeTransposeConv3, 256, "Concat3") # Concat has 512 features maps
    Out3_3 = PTF.ConvLayer(Concat3, 512, 256, ks, "Conv3_3", padding)
    Out3_4 = PTF.ConvLayer(Out3_3, 256, 256, ks, "Conv3_4", padding)
    ImageSizeTransposeConv2 = (ImageSizeTransposeConv3 - 4) * 2
    TranCon2 = PTF.TransposeConvLayer(Out3_4, 256, 128, ImageSizeTransposeConv3, ks, "TransposeConv2", "VALID")

    ####
    Concat2 = PTF.CropAndMerge(Out2_2, TranCon2, ImageSize2_2, ImageSizeTransposeConv2, 128, "Concat2") # Concat has 256 features maps
    Out2_3 = PTF.ConvLayer(Concat2, 256, 128, ks, "Conv2_3", padding)
    Out2_4 = PTF.ConvLayer(Out2_3, 128, 128, ks, "Conv2_4", padding)
    ImageSizeTransposeConv1 = (ImageSizeTransposeConv2 - 4) * 2
    TranCon1 = PTF.TransposeConvLayer(Out2_4, 128, 64, ImageSizeTransposeConv2, ks, "TransposeConv1", "VALID")

    ####
    Concat1 = PTF.CropAndMerge(Out1_2, TranCon1, ImageSize1_2, ImageSizeTransposeConv1, 64, "Concat2") # Concat has 128 features maps
    Out1_3 = PTF.ConvLayer(Concat1, 128, 64, ks, "Conv1_3", padding)
    Out1_4 = PTF.ConvLayer(Out1_3, 64, 64, ks, "Conv1_4", padding)

    logits = PTF.ConvLayerWithoutRelu(Out1_4, 64, 2, 1, "BeforeLoss", padding)

    return logits, Input, Label


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SAVE_DIR + '/test')


    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(SAVE_DIR)
    writer.add_graph(sess.graph)


    loss, optimizer, accuracy, F1, recall, precision, Inputs, Labels = Training(LEARNING_RATE, WIDTH + 184, WIDTH)


    for i in range(N_ITER_MAX):
        if i % N_RECORD == 0:
            s, acc = sess.run([merged_summary, accuracy], feed_dict=FeedDict(False, Inputs, Labels))
            test_writer.add_summary(s, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            s, _ = sess.run([merged_summary, loss], feed_dict=FeedDict(True, Inputs, Labels))
            train_writer.add_summary(s, i)
