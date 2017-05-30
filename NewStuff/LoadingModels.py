
import tensorflow as tf 
from UNet import UNetModel, FeedDict
from DataGen2 import DataGen, ListTransform
import BasicNetTF as PTF
from scipy.misc import imsave
import os
import numpy as np
import pdb

CUDA_NODE = 0
LOAD_DIR = "/share/data40T_v2/Peter/tmp/UNet/DecayLR10/test/model.ckpt-1000.meta"
HEIGHT = 212 
WIDTH = 212
CROP = 4
PATH = '/share/data40T_v2/Peter/Data/ToAnnotate'
BATCH_SIZE = 1
S = True
MEAN = np.array([104.00699, 116.66877, 122.67892])



os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_NODE)


transform_list, transform_list_test = ListTransform()
DG_TRAIN = DataGen(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                   transforms=transform_list, UNet=True)
DG_TEST  = DataGen(PATH, split="test", crop = CROP, size=(HEIGHT, WIDTH), 
                   transforms=transform_list_test, UNet=True)



KeepProbability = tf.placeholder(tf.float32, name="keep_probabilty")
PhaseTrain = tf.placeholder(tf.bool, name='phase_train')
Input, Label = PTF.InputLayer(WIDTH + 184, WIDTH, 3, PhaseTrain, S, BS=BATCH_SIZE, Weight=False)
Predicted, Logits = UNetModel(Input, PhaseTrain, WIDTH + 184, WIDTH, KeepProbability)

saver = tf.train.import_meta_graph(LOAD_DIR)


with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(LOAD_DIR)))
    sess.run(tf.global_variables_initializer())
    print("Model restored.")
    for i in range(DG_TEST.length):
        predic, logits, input_out = sess.run([Predicted, Logits, Input], feed_dict=FeedDict(False, Input, Label, PhaseTrain, DGTrain=DG_TRAIN, DGTest=DG_TEST, 
                                                          BatchSize=BATCH_SIZE, Width=WIDTH, Height=HEIGHT,
                                                          Mean=MEAN, Dim=3))   
        #pdb.set_trace()
        imsave('/share/data40T_v2/Peter/tmp/Pics/pred_{}.png'.format(i), predic[0,:,:,0])
       # pdb.set_trace()
        imsave('/share/data40T_v2/Peter/tmp/Pics/pred_{}_true.png'.format(i), input_out[0,92:-92,92:-92,:] + MEAN)

