from UNetObject import UNet
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform
import math
import pdb

class UNetBatchNorm(UNet):

    def conv_layer_f(self, i_layer, w_var, scope_name, strides=[1,1,1,1], padding="VALID"):
        with tf.name_scope(scope_name):
            conv = tf.nn.conv2d(i_layer, w_var, strides=strides, padding=padding)
            n_out = w_var.shape[3].value
            BN = self.BatchNorm(conv, n_out, self.is_training)
            #BN = conv
            return BN


    def init_vars(self):

        #### had to add is_training, self.reuse
        self.is_training = tf.placeholder(tf.bool)
        ####

        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()
        n_features = 16

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

    def Validation(self, DG_TEST, step):
        
        n_test = DG_TEST.length
        n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

        l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.


        for i in range(n_batch):
            Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval,
                         self.is_training: False}
            l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp = self.sess.run([self.loss, self.accuracy, self.F1, self.recall, self.precision, self.MeanAcc], feed_dict=feed_dict)
            l += l_tmp
            acc += acc_tmp
            F1 += F1_tmp
            recall += recall_tmp
            precision += precision_tmp
            meanacc += meanacc_tmp

        l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / n_batch


        print('  Validation loss: %.1f' % l)
        print('       Accuracy: %1.f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % (acc * 100, meanacc * 100, recall * 100, precision * 100, F1 * 100))
        self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)

    def train(self, DGTrain, DGTest, saver=True):

        epoch = DGTrain.length

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model(trainable_var)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        tf.global_variables_initializer().run()

        summary_writer = tf.summary.FileWriter(self.LOG, graph=self.sess.graph)
        merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        
        # for i in range(Xval.shape[0]):
        #     imsave("/tmp/image_{}.png".format(i), Xval[i])
        #     imsave("/tmp/label_{}.png".format(i), Yval[i,:,:,0])



        for step in range(steps):
            batch_data, batch_labels = DGTrain.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: batch_data,
                         self.train_labels_node: batch_labels,
                         self.is_training: True}

            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, merged_summary],
                        feed_dict=feed_dict)

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                summary_writer.add_summary(s, step)                
                error, acc, acc1, recall, prec, f1 = self.error_rate(predictions, batch_labels, step)
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % 
                      (l, acc, acc1, recall, prec, f1))
                self.Validation(DGTest, step)



if __name__== "__main__":

    CUDA_NODE = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)

    
    N_ITER_MAX = 2000
    N_TRAIN_SAVE = 100
    N_TEST_SAVE = 100
    LEARNING_RATE = 0.0001
    MEAN = np.array([122.67892, 116.66877 ,104.00699])
    HEIGHT = 212
    WIDTH = 212
    CROP = 4
    PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
#    PATH = '/home/pnaylor/Documents/Data/ToAnnotate'
#    PATH = "/data/users/pnaylor/Bureau/ToAnnotate"
#    PATH = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotate"

    BATCH_SIZE = 64
    LRSTEP = "epoch/2"
    SUMMARY = True
    S = SUMMARY
    SAVE_DIR = "/tmp/object/unet/bn/{}".format(LEARNING_RATE)

    transform_list, transform_list_test = ListTransform()
    DG_TRAIN = DataGenRandomT(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True)
    N_ITER_MAX = 200 * DG_TRAIN.length // BATCH_SIZE
    DG_TEST  = DataGenRandomT(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True)

    model = UNetBatchNorm(LEARNING_RATE=LEARNING_RATE,
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
