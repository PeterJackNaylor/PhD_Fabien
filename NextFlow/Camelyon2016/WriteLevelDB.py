import leveldb
import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
from optparse import OptionParser
import os
import cPickle as pkl
import pdb

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--lmdb_file", dest="lmdb_file",
                      help="leveldb_file name")

    parser.add_option("--bs", type="int", dest="bs",
                      help="batch size")
    parser.add_option("--datagen", dest="dg",
                      help="datagenerator")
    (options, args) = parser.parse_args()



    lmdb_file = options.lmdb_file
    batch_size = options.bs

    # create the leveldb file
   # db = leveldb.LevelDB(leveldb_file)
   # batch = leveldb.WriteBatch()
    
    lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin(write=True)

    datum = caffe_pb2.Datum()
    dg = pkl.load(open(options.dg, "rb"))
    key = 0
    key = dg.NextKeyRandList(key)
    pdb.set_trace()
    item_id = -1

    for x in range(dg.length):
        if x == dg.length - 1:
            pdb.set_trace()
        item_id += 1

        data, label = dg[key]

        data = data.transpose((2,0,1))

        key = dg.NextKeyRandList(key) 
        #prepare the data and label
        #data = ... #CxHxW array, uint8 or float
        #label = ... #int number

        # save in datum
        datum = caffe.io.array_to_datum(data, label)
        keystr = '{:0>8d}'.format(item_id)
        #batch.Put( keystr, datum.SerializeToString() )
        lmdb_txn.put( keystr, datum.SerializeToString() )

        # write batch
        if(item_id + 1) % batch_size == 0:
            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)
            print (item_id + 1)
            #db.Write(batch, sync=True)
            #batch = leveldb.WriteBatch()
            #print (item_id + 1)

    # write last batch
    if (item_id+1) % batch_size != 0:
        lmdb_txn.commit()
        print 'last batch'
        print (item_id + 1)
        #db.Write(batch, sync=True)
        #print 'last batch'
        #print (item_id + 1)
