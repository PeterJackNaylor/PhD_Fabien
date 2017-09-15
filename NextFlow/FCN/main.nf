#!/usr/bin/env nextflow

/// nextflow HEAD.nf -profile GPU --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter --cellcogn /share/data40T_v2/Peter/Data/CellCognition/ -resume

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "/data/users/pnaylor/Bureau/CellCognition"
params.epoch = 30

TENSORBOARD_BIN_32 = file(params.image_dir + '/tensorboard_fcn_bin_32')
TENSORBOARD_BIN_16 = file(params.image_dir + '/tensorboard_fcn_bin_16')
TENSORBOARD_BIN_8 = file(params.image_dir + '/tensorboard_fcn_bin_8')
TENSORBOARD_MULTI = file(params.image_dir + '/tensorboard_fcn_multi')
RESULTS_DIR = file(params.image_dir + '/results')

TFRECORDS = file(params.python_dir + '/Data/CreateTFRecords.py')
FCN32TRAIN = file("FCN32Train.py")
FCN32TEST = file('FCN32Test.py')
FCN16TRAIN = file("FCN16Train.py")
FCN16TEST = file('FCN16Test.py')
FCN8TRAIN = file("FCN8Train.py")
FCN8TEST = file('FCN8Test.py')
FCN8MULTITRAIN = file("FCN8MultiTrain.py")
FCN8MULTITEST = file('FCN8MultiTest.py')

BinToColorPy = file(params.python_dir + '/PrepareData/XmlParsing.py')
SlideName = file(params.python_dir + '/PrepareData/EverythingExceptColor.py')
CELLCOG_classif = file(params.cellcogn + '/classifier_January2017')
CELLCOG_folder = file(params.cellcogn + '/Fabien')

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
CHECKPOINT_VGG = file(params.image_dir + "/pretrained/vgg_16.ckpt")
IMAGE_SIZE = 224

NPRINT = 100
ITERTEST = 24

ITER32 = 5400
ITER16 = 5400
ITER8 = 5400
ITER8MULTI = 5400

LEARNING_RATE_32 = [0.001, 0.0001, 0.00001, 0.000001]
LEARNING_RATE_16 = [0.0001, 0.00001, 0.000001, 0.0000001]
LEARNING_RATE_8 = [0.00001, 0.000001, 0.0000001, 0.00000001]
LEARNING_RATE_8 = [0.00001, 0.000001, 0.0000001, 0.00000001]

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
    python $py --output TestBin.tfrecords --path $path --crop 4 --no-UNet --size $i_s --seed 42 --epoch 1 --type Normal --test
    """
}

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
    each lr from LEARNING_RATE_32
    val np from NPRINT
    val iter from ITER32
    output:
    file "log__fcn32__*" into RESULTS_32
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

process FCN32_testing {
    input:
    file py from FCN32TEST
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

    publishDir TENSORBOARD_BIN_32, mode: "copy", overwrite: false

    input:
    file _ from RES32 .toList()
    file __ from CHECKPOINT_32_1 .toList()
    file ___ from RESULTS_32 .toList()
    output:
    file "bestmodel" into STARTING_16
    file "FCN32_results.csv" into RES32_table
//    file "log__fcn32__*" into PUBLISH32
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

process FCN16 {

    clusterOptions = "-S /bin/bash"
    maxForks = 2

    input:
    file py from FCN16TRAIN
    val home from params.home
    file cp from STARTING_16
    file tfr from TrainRECORDBIN
    val i_s from IMAGE_SIZE
    each lr from LEARNING_RATE_16
    val np from NPRINT
    val iter from ITER16
    output:
    file "log__fcn16__*" into RESULTS_16
    file "model__*__fcn16s.ckpt.*" into CHECKPOINT_16
    file "model__*__fcn16s.ckpt.*" into CHECKPOINT_16_1 mode flatten
    
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint $cp --checksavedir . --tf_records $tfr --log . --image_size $i_s --labels 2 --lr $lr --n_print $np --iter $iter
    """
}

process FCN16_testing {
    input:
    file py from FCN16TEST
    file cp from CHECKPOINT_16
    file tfr from TestRECORDBIN
    val iter from ITERTEST
    val home from params.home
    output:
    file "*.csv" into RES16

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    ###PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records $tfr --labels 2 --iter $iter
    """
}

process RegroupFCN16_results {

    publishDir TENSORBOARD_BIN_16, mode: "copy", overwrite: false

    input:
    file _ from RES16 .toList()
    file __ from CHECKPOINT_16_1 .toList()
    file ___ from RESULTS_16 .toList()

    output:
    file "bestmodel" into STARTING_8
    file "FCN16_results.csv" into RES16_table
//    file "log__fcn16__*" into PUBLISH16

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
    table.to_csv('FCN16_results.csv')
    
    TOMOVE = glob("model__{}__fcn16s.ckpt*".format(best_index))
    for file in TOMOVE:
        os.rename(file, os.path.join("bestmodel", file))
    """
}

process FCN8 {

    clusterOptions = "-S /bin/bash"
//   publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file py from FCN8TRAIN
    val home from params.home
    file cp from STARTING_8
    file tfr from TrainRECORDBIN
    val i_s from IMAGE_SIZE
    each lr from LEARNING_RATE_8
    val np from NPRINT
    val iter from ITER8
    output:
    file "log__fcn8__*" into RESULTS_8
    file "model__*__fcn8s.ckpt.*" into CHECKPOINT_8
    file "model__*__fcn8s.ckpt.*" into CHECKPOINT_8_1 mode flatten
    
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint $cp --checksavedir . --tf_records $tfr --log . --image_size $i_s --labels 2 --lr $lr --n_print $np --iter $iter
    """
}

process FCN8_testing {
    input:
    file py from FCN8TEST
    file cp from CHECKPOINT_8
    file tfr from TestRECORDBIN
    val iter from ITERTEST
    val home from params.home
    output:
    file "*.csv" into RES8

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    ###PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records $tfr --labels 2 --iter $iter
    """
}

process RegroupFCN8_results {

    publishDir TENSORBOARD_BIN_8, mode: "copy", overwrite: false


    input:
    file _ from RES8 .toList()
    file __ from CHECKPOINT_8_1 .toList()
    file ___ from RESULTS_8 .toList()

    output:
    file "bestmodel" into STARTING_MULTI
    file "FCN8_results.csv" into RES8_table
//    file "log__fcn8__*" into PUBLISH8

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
    table.to_csv('FCN8_results.csv')
    
    TOMOVE = glob("model__{}__fcn8s.ckpt*".format(best_index))
    for file in TOMOVE:
        os.rename(file, os.path.join("bestmodel", file))
    """
}

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
    file "./ToAnnotateColor" into TAC, TAC2, TAC3, TAC4

    """
    python $py --a $classifier --c $cellcog_folder --o ./ToAnnotateColor/ --d ./Diff/
    python $py2 -i $toannotate --o_c $cellcog_folder --o_b ./ToAnnotateBinary/
    cp -r ./ToAnnotateBinary/Slide_* ./ToAnnotateColor/
    """
}

process CreateMultiRecordsTrain {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from TAC

    output:
    file "MultiFCNTrain.tfrecords" into DATAQUEUE_MULTI
    """

    python $py --output MultiFCNTrain.tfrecords --path $path --crop 4 --no-UNet --size 224 --seed 42 --epoch $epoch --type ReducedClass --train
    """
}

process CreateMultiRecordsTest {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from TAC2

    output:
    file "MultiFCNTest.tfrecords" into DATAQUEUE_MULTI_TEST
    """

    python $py --output MultiFCNTest.tfrecords --path $path --crop 4 --no-UNet --size 224 --seed 42 --epoch 1 --type ReducedClass --test
    """
}

process FCN8_Multi {

    clusterOptions = "-S /bin/bash"
    maxForks = 2

    input:
    file py from FCN8MULTITRAIN
    val home from params.home
    file cp from STARTING_MULTI
    file tfr from DATAQUEUE_MULTI
    val i_s from IMAGE_SIZE
    each lr from LEARNING_RATE_8MULTI
    val np from NPRINT
    val iter from ITER8MULTI
    output:
    file "log__fcn8Multi__*" into RESULTS_8MULTI
    file "model__*__fcn8Multis.ckpt.*" into CHECKPOINT_8MULTI
    file "model__*__fcn8Multis.ckpt.*" into CHECKPOINT_8MULTI_1 mode flatten
    
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint $cp --checksavedir . --tf_records $tfr --log . --image_size $i_s --labels 6 --lr $lr --n_print $np --iter $iter
    """
}

process FCN8_Multi_testing {
    input:
    file py from FCN8MULTITEST
    file cp from CHECKPOINT_8MULTI
    file tfr from DATAQUEUE_MULTI_TEST
    val iter from ITERTEST
    val home from params.home
    output:
    file "*.csv" into RES8MULTI

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    ###PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records $tfr --labels 6 --iter $iter
    """
}

process RegroupFCN8Multi_results {

    publishDir TENSORBOARD_MULTI, mode: "copy", overwrite: false


    input:
    file _ from RES8MULTI .toList()
    file __ from CHECKPOINT_8MULTI_1 .toList()
    file ___ from RESULTS_8MULTI .toList()

    output:
    file "bestmodel" into BEST_MULTI_MODEL
    file "FCN8Multi_results.csv" into RES8Multi_table
//    file "log__fcn8Multi__*" into PUBLISH8Multi

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
    table.to_csv('FCN8Multi_results.csv')
    
    TOMOVE = glob("model__{}__fcn8Multis.ckpt*".format(best_index))
    for file in TOMOVE:
        os.rename(file, os.path.join("bestmodel", file))
    """
}
