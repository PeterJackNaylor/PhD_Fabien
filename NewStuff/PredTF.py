import tensorflow as tf
import pdb
import os
import numpy as np



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x[:,:,0] / e_x.sum(axis=2)


def PredImageFromNetTF(model, load_meta, window, MEAN_FILE="mean_file.npy"):
    """
    net: is a three items 
    """
    input_var = model.input_node
    logits = model.logits
    predictions = model.predictions
    #pdb.set_trace()
    ### name bug when saving something to do with exponential moving average
    ### see for more details https://github.com/tensorflow/tensorflow/issues/2768
    names_to_vars = {v.op.name: v for v in tf.global_variables()}
    for i in range(1, 5):
        for j in range(1, 5):
            exp_mean = names_to_vars["conv{}_{}/bn/moments/Squeeze/ExponentialMovingAverage".format(i, j)]
            names_to_vars["conv{}_{}/bn/moments/moments_1/mean/ExponentialMovingAverage".format(i, j)] = exp_mean
            del names_to_vars["conv{}_{}/bn/moments/Squeeze/ExponentialMovingAverage".format(i, j)]

            exp_var = names_to_vars["conv{}_{}/bn/moments/Squeeze_1/ExponentialMovingAverage".format(i, j)]
            names_to_vars["conv{}_{}/bn/moments/moments_1/variance/ExponentialMovingAverage".format(i, j)] = exp_var
            del names_to_vars["conv{}_{}/bn/moments/Squeeze_1/ExponentialMovingAverage".format(i, j)]
    
    for i in range(5, 6):
        for j in range(1, 3):
            exp_mean = names_to_vars["conv{}_{}/bn/moments/Squeeze/ExponentialMovingAverage".format(i, j)]
            names_to_vars["conv{}_{}/bn/moments/moments_1/mean/ExponentialMovingAverage".format(i, j)] = exp_mean
            del names_to_vars["conv{}_{}/bn/moments/Squeeze/ExponentialMovingAverage".format(i, j)]

            exp_var = names_to_vars["conv{}_{}/bn/moments/Squeeze_1/ExponentialMovingAverage".format(i, j)]
            names_to_vars["conv{}_{}/bn/moments/moments_1/variance/ExponentialMovingAverage".format(i, j)] = exp_var
            del names_to_vars["conv{}_{}/bn/moments/Squeeze_1/ExponentialMovingAverage".format(i, j)]
    
    exp_mean = names_to_vars["logits/bn/moments/Squeeze/ExponentialMovingAverage"]
    names_to_vars["logits/bn/moments/moments_1/mean/ExponentialMovingAverage"] = exp_mean
    del names_to_vars["logits/bn/moments/Squeeze/ExponentialMovingAverage"]

    exp_var = names_to_vars["logits/bn/moments/Squeeze_1/ExponentialMovingAverage"]
    names_to_vars["logits/bn/moments/moments_1/variance/ExponentialMovingAverage"] = exp_var
    del names_to_vars["logits/bn/moments/Squeeze_1/ExponentialMovingAverage"]

    saver = tf.train.Saver(var_list=names_to_vars)
    if MEAN_FILE is not None:
        window = window - np.load(MEAN_FILE)
    if hasattr(model, "is_training"):
        DIC = {input_var: np.expand_dims(window, axis=0),
                model.is_training: False}
    else:
        DIC = {input_var: np.expand_dims(window, axis=0)}
#    saver = tf.train.import_meta_graph(load_meta)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(load_meta))

    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            print("Model restored...")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Mayde"
        # Restore variables from disk.
#        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(load_meta)))
#        sess.run(tf.global_variables_initializer())
        bin, logits = sess.run([predictions, logits], feed_dict=DIC)
        prob = softmax(logits[0])

    return prob, bin[0]

