#!/usr/bin/env nextflow

/// nextflow HEAD.nf -profile GPU --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter --cellcogn /share/data40T_v2/Peter/Data/CellCognition/ -resume

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "/data/users/pnaylor/Bureau/CellCognition"


PY = file(params.python_dir + '/Nets/UNetMultiClass_v2.py')


TENSORBOARD_BIN = file(params.image_dir + '/tensorboard_fcn_bin')
TENSORBOARD_MULTI = file(params.image_dir + '/tensorboard_fcn_multi')


TFRECORDS = file(params.python_dir + '/Data/CreateTFRecords.py')
params.epoch = 10
IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
IMAGE_SIZE = 224

process CreateBWRecords {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZE

    output:
    file "TrainBin.tfrecords" into TrainRECORDBIN
    """
    ###PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --output TrainBin.tfrecords --path $path --crop 4 --no-UNet --size $i_s --seed 42 --epoch $epoch --type Normal --train
    """
}

process CreateBWTestRecords {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZE

    output:
    file "TestBin.tfrecords" into TestRECORDBIN
    """
    ##PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --output TestBin.tfrecords --path $path --crop 4 --no-UNet --size $i_s --seed 42 --epoch $epoch --type Normal --test
    """
}

FCN32TRAIN = file("FCN32Train.py")
CHECKPOINT_VGG = file(params.image_dir + "/pretrained/vgg_16.ckpt")
LEARNING_RATE = [0.001, 0.0001, 0.00001, 0.000001]

NPRINT = 100
ITER32 = 200

process FCN32 {

    clusterOptions = "-S /bin/bash"
//   publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file py from FCN32TRAIN
    val home from params.home
    file cp from CHECKPOINT_VGG
    file tfr from TrainRECORDBIN
    val i_s from IMAGE_SIZE
    each lr from LEARNING_RATE
    val np from NPRINT
    val iter from ITER32
    output:
    file "log__fcn32__*" into RESULTS 
    file "model__*__fcn32s.ckpt.*" into CHECKPOINT_32
    file "model__*__fcn32s.ckpt.*" into CHECKPOINT_32_1 mode flatten
    
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint $cp --checksavedir . --tf_records $tfr --log . --image_size $i_s --labels 2 --lr $lr --n_print $np --iter $iter
    """
}

FCNTEST = file('FCN32Test.py')
ITERTEST = 24

process FCN32_testing {
    input:
    file py from FCNTEST
    file cp from CHECKPOINT_32
    file tfr from TestRECORDBIN
    val iter from ITERTEST
    val home from params.home
    output:
    file "*.csv" into RES32

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    ###PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records $tfr --labels 2 --iter $iter
    """

}

process RegroupFCN32_results {

    input:
    file _ from RES32 .toList()
    file __ from CHECKPOINT_32_1 .toList()
    output:
    file "bestmodel" into STARTING_16
    file "FCN32_results.csv" into RES32_table
    """
    #!/usr/bin/env python
    import os
    os.mkdir('bestmodel')
    import pandas as pd
    import pdb
    from glob import glob
    CSV = glob('*.csv')
    df_list = []
    for f in CSV:
        df = pd.read_csv(f, index_col=0)
        df.index = [f[0:-4].split('__')[1]]
        df_list.append(df)
    table = pd.concat(df_list)

    best_index = table['F1'].argmax()
    table.to_csv('FCN32_results.csv')
    
    TOMOVE = glob("model__{}__fcn32s.ckpt*".format(best_index))
    for file in TOMOVE:
        os.rename(file, os.path.join("bestmodel", file))
    """

}

BinToColorPy = file(params.python_dir + '/PrepareData/XmlParsing.py')
SlideName = file(params.python_dir + '/PrepareData/EverythingExceptColor.py')
CELLCOG_classif = file(params.cellcogn + '/classifier_January2017')
CELLCOG_folder = file(params.cellcogn + '/Fabien')

process BinToColor {
    executor 'local'
    clusterOptions = "-S /bin/bash"

    input:
    file py from BinToColorPy
    file py2 from SlideName
    file toannotate from IMAGE_FOLD
    file classifier from CELLCOG_classif
    file cellcog_folder from CELLCOG_folder
    output:
    file "./ToAnnotateColor" into ToAnnotateColor, ToAnnotateColor2

    """
    python $py --a $classifier --c $cellcog_folder --o ./ToAnnotateColor/ --d ./Diff/
    python $py2 -i $toannotate --o_c $cellcog_folder --o_b ./ToAnnotateBinary/
    cp -r ./ToAnnotateBinary/Slide_* ./ToAnnotateColor/
    """
}
/*
process CreateMultiRecords {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from ToAnnotateColor

    output:
    file "UNetRecords.tfrecords" into DATAQUEUE
    """

    python $py --output TrainMulti.tfrecords --path $path --crop 4 --no-UNet --size 224 --seed 42 --epoch $epoch --type ReducedClass
    """
}


*/
