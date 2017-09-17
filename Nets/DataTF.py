from ObjectOriented import ConvolutionalNeuralNetwork


from Data.DataGenRandomT import DataGenRandomT
from UsefulFunctions.ImageTransf import ListTransform
import os
import tensorflow as tf
from datetime import datetime
import skimage.io as io
import numpy as np
import math


class DataReader(ConvolutionalNeuralNetwork):
    def __init__(
        self,
        TF_RECORDS,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_LABELS=10,
        NUM_CHANNELS=1,
        NUM_TEST=10000,
        STEPS=2000,
        LRSTEP=200,
        DECAY_EMA=0.9999,
        N_PRINT = 100,
        LOG="/tmp/net",
        SEED=42,
        DEBUG=True,
        WEIGHT_DECAY=0.00005,
        LOSS_FUNC=tf.nn.l2_loss,
        N_FEATURES=16,
        N_EPOCH=1,
        N_THREADS=1,
        MEAN_FILE=None,
        DROPOUT=0.5):

        self.LEARNING_RATE = LEARNING_RATE
        self.K = K
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_LABELS = NUM_LABELS
        self.NUM_CHANNELS = NUM_CHANNELS
        self.N_FEATURES = N_FEATURES
#        self.NUM_TEST = NUM_TEST
        self.STEPS = STEPS
        self.N_PRINT = N_PRINT
        self.LRSTEP = LRSTEP
        self.DECAY_EMA = DECAY_EMA
        self.LOG = LOG
        self.SEED = SEED
        self.N_EPOCH = N_EPOCH
        self.N_THREADS = N_THREADS
        self.DROPOUT = DROPOUT
        if MEAN_FILE is not None:
            MEAN_ARRAY = tf.constant(np.load(MEAN_FILE), dtype=tf.float32) # (3)
            self.MEAN_ARRAY = tf.reshape(MEAN_ARRAY, [1, 1, 3])
            self.SUB_MEAN = True
        else:
            self.SUB_MEAN = False

        self.sess = tf.InteractiveSession()

        self.sess.as_default()
        
        self.var_to_reg = []
        self.var_to_sum = []
        self.TF_RECORDS = TF_RECORDS
        self.init_queue(TF_RECORDS)

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()
        self.Saver()
        self.DEBUG = DEBUG
        self.loss_func = LOSS_FUNC
        self.weight_decay = WEIGHT_DECAY

    def init_queue(self, tfrecords_filename):
        self.filename_queue = tf.train.string_input_producer(
                              [tfrecords_filename], num_epochs=10)
        with tf.device('/cpu:0'):
            self.image, self.annotation = read_and_decode(self.filename_queue, 
                                                          self.IMAGE_SIZE[0], 
                                                          self.IMAGE_SIZE[1],
                                                          self.BATCH_SIZE,
                                                          self.N_THREADS)
        print("Queue initialized")
    def input_node_f(self):
        if self.SUB_MEAN:
            self.images_queue = self.image - self.MEAN_ARRAY
        else:
            self.images_queue = self.image
        self.image_PH = tf.placeholder_with_default(self.images_queue, shape=[self.BATCH_SIZE,
                                                                              self.IMAGE_SIZE[0], 
                                                                              self.IMAGE_SIZE[1],
                                                                              3])
        return self.image_PH
    def label_node_f(self):
        self.labels_queue = self.annotation
        self.labels_PH = tf.placeholder_with_default(self.labels_queue, shape=[self.BATCH_SIZE,
                                                                          self.IMAGE_SIZE[0], 
                                                                          self.IMAGE_SIZE[1],
                                                                          1])

        return self.labels_PH
    def Validation(self, DG_TEST, step):
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(math.ceil(float(n_test) / self.BATCH_SIZE)) 

            l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.

            for i in range(n_batch):
                Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval}
                l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp, pred, s = self.sess.run([self.loss, 
                                                                                            self.accuracy, self.F1,
                                                                                            self.recall, self.precision,
                                                                                            self.MeanAcc, self.predictions,
                                                                                            self.merged_summary], feed_dict=feed_dict)
                l += l_tmp
                acc += acc_tmp
                F1 += F1_tmp
                recall += recall_tmp
                precision += precision_tmp
                meanacc += meanacc_tmp

            l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / n_batch

            summary = tf.Summary()
            summary.value.add(tag="TestMan/Accuracy", simple_value=acc)
            summary.value.add(tag="TestMan/Loss", simple_value=l)
            summary.value.add(tag="TestMan/F1", simple_value=F1)
            summary.value.add(tag="TestMan/Recall", simple_value=recall)
            summary.value.add(tag="TestMan/Precision", simple_value=precision)
            summary.value.add(tag="TestMan/Performance", simple_value=meanacc)
            self.summary_test_writer.add_summary(summary, step) 
            self.summary_test_writer.add_summary(s, step) 
            print('  Validation loss: %.1f' % l)
            print('       Accuracy: %1.f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % (acc * 100, meanacc * 100, recall * 100, precision * 100, F1 * 100))
            self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)

    def train(self, DG_TEST=None):

        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model()
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(steps):      
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                error, acc, acc1, recall, prec, f1 = self.error_rate(predictions, batch_labels, step)
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % 
                     (l, acc, acc1, recall, prec, f1))
                self.Validation(DG_TEST, step)
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':

    TRAIN_TF = True
    CUDA_NODE = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)

    SAVE_DIR = "/tmp/object/CheckQueue2"
    N_ITER_MAX = 2000
    N_TRAIN_SAVE = 42
    N_TEST_SAVE = 100
    LEARNING_RATE = 0.001
    SIZE = (224, 224)
    CROP = 4
    PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
    PATH = '/home/pnaylor/Documents/Data/ToAnnotate'
    PATH = "/data/users/pnaylor/Bureau/ToAnnotate"
    PATH = "/Users/naylorpeter/Documents/Histopathologie/ToAnnotate/ToAnnotate"
    OUTNAME = 'firsttry.tfrecords'
    SPLIT = 'train'
    BATCH_SIZE = 2
    LRSTEP = "10epoch"
    SUMMARY = True
    N_EPOCH = 1
    SEED = 42
    S = SUMMARY
    TEST_PATIENT = ["141549", "162438"]
    MEAN_FILE = "mean_file.npy"
    UNET = False
    transform_list, transform_list_test = ListTransform()
    TRANSFORM_LIST = transform_list

    DG = DataGenRandomT(PATH, split=SPLIT, crop=CROP, size=SIZE,
                        transforms=TRANSFORM_LIST, UNet=UNET,
                        mean_file=None, seed_=SEED)
    DG.SetPatient(TEST_PATIENT)

    DG_TEST = DataGenRandomT(PATH, split="test", crop=CROP, size=SIZE,
                        transforms=transform_list_test, UNet=UNET,
                        mean_file=MEAN_FILE, seed_=SEED)
    DG_TEST.SetPatient(TEST_PATIENT)
    N_ITER_MAX = N_EPOCH * DG.length // BATCH_SIZE

    if TRAIN_TF:
        model = DataReader(OUTNAME,LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_LABELS=2,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=100,
                                       MEAN_FILE=MEAN_FILE)

        model.train(DG_TEST)
        
