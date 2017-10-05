from UNetDistance import UNetDistance
import tensorflow as tf
import numpy as np
from Data.CreateTFRecords import read_and_decode
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from datetime import datetime
from optparse import OptionParser
from UsefulFunctions.ImageTransf import ListTransform
from Data.DataGenClass import DataGen3, DataGenMulti, DataGen3reduce


class Test(UNetDistance):
    def __init__(
        self,
        TF_RECORDS,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
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
        self.WritteSummaryImages()
    def init_queue(self, tfrecords_filename):
        self.filename_queue = tf.train.string_input_producer(
                              [tfrecords_filename], num_epochs=10)
        with tf.device('/cpu:0'):
            self.image, self.annotation = read_and_decode(self.filename_queue, 
                                                          self.IMAGE_SIZE[0], 
                                                          self.IMAGE_SIZE[1],
                                                          self.BATCH_SIZE,
                                                          self.N_THREADS,
                                                          True)
            self.annotation = tf.divide(self.annotation, 255.)
        print("Queue initialized")
    def init_vars(self):

        #### had to add is_training, self.reuse

        self.is_training = tf.placeholder_with_default(True, shape=[])
        ####

        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()
        self.keep_prob = tf.Variable(self.DROPOUT, name="dropout_prob")

        print('Model variables initialised')
    def init_model_architecture(self):

        print('Model architecture initialised')
    def init_training_graph(self):

        tf.global_variables_initializer().run()

        print('Computational graph initialised')
    def WritteSummaryImages(self):
        Size1 = tf.shape(self.input_node)[1]
        Size_to_be = tf.cast(Size1, tf.int32) - 92
        self.size = Size_to_be
        self.crop_input_node = tf.slice(self.input_node, [0, 92, 92, 0], [-1, 212, 212, -1])

        tf.summary.image("Input", self.crop_input_node, max_outputs=4)
        tf.summary.image("Label", self.train_labels_node, max_outputs=4)

    def train(self, DGTest):

        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH

        trainable_var = tf.trainable_variables()
        
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

        for step in range(1):      
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            image, annotation, image_mean, crop, size, s = self.sess.run(
                        [self.image, self.annotation, self.images_queue, self.crop_input_node, self.size, self.merged_summary])
            self.summary_writer.add_summary(s, step)                
                
        coord.request_stop()
        coord.join(threads)
        return image, annotation, image_mean, crop, size

if __name__== "__main__":

    parser = OptionParser()

#    parser.add_option("--gpu", dest="gpu", default="0", type="string",
#                      help="Input file (raw data)")
    parser.add_option("--tf_record", dest="TFRecord", type="string",
                      help="Where to find the TFrecord file")

    parser.add_option("--path", dest="path", type="string",
                      help="Where to collect the patches")

    parser.add_option("--log", dest="log",
                      help="log dir")

    parser.add_option("--learning_rate", dest="lr", type="float",
                      help="learning_rate")

    parser.add_option("--batch_size", dest="bs", type="int",
                      help="batch size")

    parser.add_option("--epoch", dest="epoch", type="int",
                      help="number of epochs")

    parser.add_option("--n_features", dest="n_features", type="int",
                      help="number of channels on first layers")

    parser.add_option("--weight_decay", dest="weight_decay", type="float",
                      help="weight decay value")

    parser.add_option("--dropout", dest="dropout", type="float",
                      default=0.5, help="dropout value to apply to the FC layers.")

    parser.add_option("--mean_file", dest="mean_file", type="str",
                      help="where to find the mean file to substract to the original image.")

    parser.add_option('--n_threads', dest="THREADS", type=int, default=100,
                      help="number of threads to use for the preprocessing.")

    (options, args) = parser.parse_args()


    TFRecord = options.TFRecord
    N_FEATURES = options.n_features
    WEIGHT_DECAY = options.weight_decay
    DROPOUT = options.dropout
    MEAN_FILE = options.mean_file 
    N_THREADS = options.THREADS



    LEARNING_RATE = options.lr
    if int(str(LEARNING_RATE)[-1]) > 7:
        lr_str = "1E-{}".format(str(LEARNING_RATE)[-1])
    else:
        lr_str = "{0:.8f}".format(LEARNING_RATE).rstrip("0")
    SAVE_DIR = options.log + "/" + "{}".format(N_FEATURES) + "_" +"{0:.8f}".format(WEIGHT_DECAY).rstrip("0") + "_" + lr_str

    
    
    HEIGHT = 224 
    WIDTH = 224
    
    
    BATCH_SIZE = options.bs
    LRSTEP = "10epoch"
    SUMMARY = True
    S = SUMMARY
    N_EPOCH = options.epoch

    PATH = options.path


    HEIGHT = 212
    WIDTH = 212
    SIZE = (HEIGHT, WIDTH)

    N_TRAIN_SAVE = 100
 
    CROP = 4


    transform_list, transform_list_test = ListTransform(n_elastic=0)

    DG_TRAIN = DataGenMulti(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, UNet=True, mean_file=None)

    test_patient = ["141549", "162438"]
    DG_TRAIN.SetPatient(test_patient)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE

    DG_TEST  = DataGenMulti(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
    DG_TEST.SetPatient(test_patient)

    model = Test(TFRecord,    LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=N_THREADS,
                                       MEAN_FILE=MEAN_FILE,
                                       DROPOUT=DROPOUT)
    Xval, Yval = DG_TEST.Batch(0, 1)
    feed_dict = {model.input_node: Xval,model.train_labels_node: Yval,model.is_training: False}
#    with tf.Session() as sess:
#        image, annotation = sess.run([model.image, model.annotation], feed_dict=feed_dict)
    image, annotation, mean, crop, size = model.train(DG_TEST)