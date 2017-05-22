# -*- coding: utf-8 -*-

import tensorflow as tf
import BasicNetTF as PTF
import numpy as np
from DataGen2 import DataGen, ListTransform
import pdb
import os
import time




CUDA_NODE = 0 
SAVE_DIR = "/share/data40T_v2/Peter/tmp/UNet/DecayLR8"
N_ITER_MAX = 1000
N_TRAIN_SAVE = 100
N_TEST_SAVE = 200
LEARNING_RATE = 0.01
MEAN = np.array([104.00699, 116.66877, 122.67892])
HEIGHT = 212 
WIDTH = 212
CROP = 4
PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
BATCH_SIZE = 6
LRSTEP = 200
SUMMARY = True
S = SUMMARY




os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)


transform_list, transform_list_test = ListTransform()
DG_TRAIN = DataGen(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                   transforms=transform_list, UNet=True)
DG_TEST  = DataGen(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                   transforms=transform_list_test, UNet=True)


def FeedDict(Train, Inputs, Labels, PhaseTrain, DGTrain=DG_TRAIN, DGTest=DG_TEST, 
              BatchSize=BATCH_SIZE, Width=WIDTH, Height=HEIGHT,
              Mean=MEAN, Dim=3):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

    ImagesBatch = np.zeros(shape=(BatchSize, Width + 184, Height + 184, Dim), dtype='float')
    LabelsBatch = np.zeros(shape=(BatchSize, Width, Height, 1), dtype='float')   

    if Train:

        for i in range(BatchSize):

            key = DGTrain.NextKeyRandList(0)
            ImagesBatch[i], LabelsBatch[i,:,:,0] = DGTrain[key]
            ImagesBatch[i] -= Mean

    else:

        for i in range(BatchSize):

            key = DGTest.NextKeyRandList(0)
            ImagesBatch[i], LabelsBatch[i,:,:,0] = DGTest[key]   
            ImagesBatch[i] -= Mean

    return {Inputs: ImagesBatch, Labels: LabelsBatch, PhaseTrain: Train}




def UNetModel(Input, PhaseTrain, ImageSizeIn, ImageSizeOut, KeepProbability):

    ks = 3
    padding = "VALID"

    Out1_1 = PTF.ConvBatchLayer(Input, 3, 64, ks, "Conv1_1", padding, PhaseTrain, S)
    Out1_2 = PTF.ConvBatchLayer(Out1_1, 64, 64, ks, "Conv1_2", padding, PhaseTrain, S)
    ImageSize1_2 = ImageSizeIn - 4
    MaxPool1 = tf.nn.max_pool(Out1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool1")

    Out2_1 = PTF.ConvBatchLayer(MaxPool1, 64, 128, ks, "Conv2_1", padding, PhaseTrain, S)
    Out2_2 = PTF.ConvBatchLayer(Out2_1, 128, 128, ks, "Conv2_2", padding, PhaseTrain, S)
    ImageSize2_2 = ImageSize1_2 / 2 - 4
    MaxPool2 = tf.nn.max_pool(Out2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool2")

    Out3_1 = PTF.ConvBatchLayer(MaxPool2, 128, 256, ks, "Conv3_1", padding, PhaseTrain, S)
    Out3_2 = PTF.ConvBatchLayer(Out3_1, 256, 256, ks, "Conv3_2", padding, PhaseTrain, S)
    ImageSize3_2 = ImageSize2_2 / 2 - 4
    MaxPool3 = tf.nn.max_pool(Out3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool3")

    Out4_1 = PTF.ConvBatchLayer(MaxPool3, 256, 512, ks, "Conv4_1", padding, PhaseTrain, S)
    Out4_2 = PTF.ConvBatchLayer(Out4_1, 512, 512, ks, "Conv4_2", padding, PhaseTrain, S)
    ImageSize4_2 = ImageSize3_2 / 2 - 4
    MaxPool4 = tf.nn.max_pool(Out4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Maxpool4")

    Out5_1 = PTF.ConvBatchLayer(MaxPool4, 512, 1024, ks, "Conv5_1", padding, PhaseTrain, S)
    Out5_2 = PTF.ConvBatchLayer(Out5_1, 1024, 1024, ks, "Conv5_2", padding, PhaseTrain, S)
    ImageSizeTransposeConv4 = (ImageSize4_2 / 2 - 4) * 2
    TranCon4 = PTF.TransposeConvLayer(Out5_2, 1024, 512, 2, "TransposeConv4", padding, PhaseTrain, S)

    #### 
    Concat4 = PTF.CropAndMerge(Out4_2, TranCon4, ImageSize4_2, ImageSizeTransposeConv4, "Concat4") # Concat has 1024 features maps
    Out4_3 = PTF.ConvBatchLayer(Concat4, 1024, 512, ks, "Conv4_3", padding, PhaseTrain, S)
    Out4_4 = PTF.ConvBatchLayer(Out4_3, 512, 512, ks, "Conv4_4", padding, PhaseTrain, S)
    ImageSizeTransposeConv3 = (ImageSizeTransposeConv4 - 4) * 2
    TranCon3 = PTF.TransposeConvLayer(Out4_4, 512, 256, 2, "TransposeConv3", padding, PhaseTrain, S)

    ####
    Concat3 = PTF.CropAndMerge(Out3_2, TranCon3, ImageSize3_2, ImageSizeTransposeConv3, "Concat3") # Concat has 512 features maps
    Out3_3 = PTF.ConvBatchLayer(Concat3, 512, 256, ks, "Conv3_3", padding, PhaseTrain, S)
    Out3_4 = PTF.ConvBatchLayer(Out3_3, 256, 256, ks, "Conv3_4", padding, PhaseTrain, S)
    ImageSizeTransposeConv2 = (ImageSizeTransposeConv3 - 4) * 2
    TranCon2 = PTF.TransposeConvLayer(Out3_4, 256, 128, 2, "TransposeConv2", padding, PhaseTrain, S)

    ####
    Concat2 = PTF.CropAndMerge(Out2_2, TranCon2, ImageSize2_2, ImageSizeTransposeConv2, "Concat2") # Concat has 256 features maps
    Out2_3 = PTF.ConvBatchLayer(Concat2, 256, 128, ks, "Conv2_3", padding, PhaseTrain, S)
    Out2_4 = PTF.ConvBatchLayer(Out2_3, 128, 128, ks, "Conv2_4", padding, PhaseTrain, S)
    ImageSizeTransposeConv1 = (ImageSizeTransposeConv2 - 4) * 2
    TranCon1 = PTF.TransposeConvLayer(Out2_4, 128, 64, 2, "TransposeConv1", padding, PhaseTrain, S)

    ####
    Concat1 = PTF.CropAndMerge(Out1_2, TranCon1, ImageSize1_2, ImageSizeTransposeConv1, "Concat2") # Concat has 128 features maps
    Out1_3 = PTF.ConvBatchLayer(Concat1, 128, 64, ks, "Conv1_3", padding, PhaseTrain, S)
    Out1_4 = PTF.ConvBatchLayer(Out1_3, 64, 64, ks, "Conv1_4", padding, PhaseTrain, S)

    logits = PTF.ConvBatchLayerWithoutRelu(Out1_4, 64, 2, 1, "BeforeLoss", padding, PhaseTrain, S)
    annotation_pred = tf.argmax(logits, dimension=3, name="Prediction")

    return tf.expand_dims(annotation_pred, dim=3), logits


def train(LossVal, VarList, LearningRate, LRUpdate):

    k = 0.96
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate =  LearningRate
    #pdb.set_trace()
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                LRUpdate, k, staircase=True)
    tf.summary.scalar('LearningRate', learning_rate)

#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#    grads = optimizer.compute_gradients(LossVal, var_list=VarList)
    optimizer = (tf.train.GradientDescentOptimizer(learning_rate).minimize(LossVal, global_step=global_step))

#    return optimizer.apply_gradients(grads)
    return optimizer

def main(argv=None):

    KeepProbability = tf.placeholder(tf.float32, name="keep_probabilty")
    PhaseTrain = tf.placeholder(tf.bool, name='phase_train')
    Input, Label = PTF.InputLayer(WIDTH + 184, WIDTH, 3, PhaseTrain, S, BS=BATCH_SIZE, Weight=False)
    

    LabelInt = tf.to_int64(Label)
    Predicted, Logits = UNetModel(Input, PhaseTrain, WIDTH + 184, WIDTH, KeepProbability)
    if S:
        tf.summary.image("pred_annotation", tf.cast(Predicted, tf.uint8), max_outputs=4)

    with tf.name_scope('Loss'):

        Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Logits,
                                                                              labels=tf.squeeze(LabelInt, squeeze_dims=[3]),
                                                                              name="entropy"))
        tf.summary.scalar('Loss', Loss)

    with tf.name_scope('Accuracy'):
 
        CorrectPrediction = tf.equal(Predicted, LabelInt)
        Accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))

        with tf.name_scope('F1'):

            with tf.name_scope('Subs'):

                TP = tf.count_nonzero(Predicted * LabelInt)
                TN = tf.count_nonzero((Predicted - 1) * (LabelInt - 1))
                FP = tf.count_nonzero(Predicted * (LabelInt - 1))
                FN = tf.count_nonzero((Predicted - 1) * LabelInt)

            with tf.name_scope('Precision'):

                precision = tf.divide(TP, tf.add(TP, FP))

            with tf.name_scope('Recall'):

                recall = tf.divide(TP, tf.add(TP, TN))

            with tf.name_scope('F1'):

                num = tf.multiply(precision, recall)
                dem = tf.add(precision, recall)
                F1 = tf.scalar_mul(2, tf.divide(num, dem))

        tf.summary.scalar('Accuracy', Accuracy)
        tf.summary.scalar('Precision', precision)
        tf.summary.scalar('Recall', recall)
        tf.summary.scalar('F1', F1)


    trainable_var = tf.trainable_variables()
    train_op = train(Loss, trainable_var, LEARNING_RATE, LRSTEP)
    merged_summary = tf.summary.merge_all()

    sess = tf.Session()

    print("Setting up Saver...")

    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SAVE_DIR + '/test')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(SAVE_DIR + '/train')

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    TIME = time.time()
    for steps in range(1, N_ITER_MAX+1):

        feed_dict_train = FeedDict(True, Input, Label, PhaseTrain)
        sess.run(train_op, feed_dict=feed_dict_train)

        if steps % N_TRAIN_SAVE == 0:
            train_loss, train_acc, train_f1, summary_str = sess.run([Loss, Accuracy, F1, merged_summary], feed_dict=feed_dict_train)
            train_writer.add_summary(summary_str, steps)
            print('Training loss {:5d}: {:.2f}, Accuracy: {:.2f}, F1: {:.2f}'.format(steps, train_loss, train_acc, train_f1))

            diff_time = time.time() - TIME
            TIME = time.time()
            print '\t%02i:%02i:%02i' % (diff_time / 3600, (diff_time % 3600) / 60, diff_time % 60)
        if steps % N_TEST_SAVE == 0:
            feed_dict_test = FeedDict(False, Input, Label, PhaseTrain)
            test_loss, test_acc, test_f1 = sess.run([Loss, Accuracy, F1], feed_dict=feed_dict_test)
            test_writer.add_summary(summary_str, steps)
            print('Validation loss {:5d}: {:.2f}, Accuracy: {:.2f}, F1: {:.2f}'.format(steps, test_loss, test_acc, test_f1))
            saver.save(sess, SAVE_DIR + '/test/' + "model.ckpt", steps)


#    loss, train_op, accuracy, F1, recall, precision, Inputs, Labels = Training(LEARNING_RATE,)


if __name__ == "__main__":
    tf.app.run()





def Training(LearningRate, ImageSizeIn, ImageSizeOut, BatchSize=BATCH_SIZE):

    
    with tf.name_scope('Metrics'):

        with tf.name_scope('CorrectPrediction'):


            correct_prediction = tf.equal(Predicted, LabelsInt)

        with tf.name_scope('Calculation'):

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('F1'):

            with tf.name_scope('Subs'):

                TP = tf.count_nonzero(Predicted * LabelsInt)
                TN = tf.count_nonzero((Predicted - 1) * (LabelsInt - 1))
                FP = tf.count_nonzero(Predicted * (LabelsInt - 1))
                FN = tf.count_nonzero((Predicted - 1) * LabelsInt)

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
        grads = optimizer.compute_gradients(Loss, var_list=trainable_var)

    tf.summary.scalar('LearningRate', LearningRate)

    return Loss, optimizer.apply_gradients(grads), accuracy, F1, recall, precision, Input, Label
