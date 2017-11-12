#!/usr/bin/env nextflow

params.image_dir = '/share/data40T_v2/Peter/Data'
params.python_dir = '/share/data40T_v2/Peter/PythonScripts/PhD_Fabien'
params.home = "/share/data40T_v2/Peter"
params.epoch = 50
IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
PY = file(params.python_dir + '/Nets/UNetBatchNorm_v2.py')
TENSORBOARD = file(params.image_dir + '/tensorboard_withmean')
MEANPY = file(params.python_dir + '/Data/MeanCalculation.py')
TFRECORDS = file('src/TFRecords.py')
PYTEST = file('src/Testing.py')
LEARNING_RATE = [0.001, 0.0001, 0.00001]
ARCH_FEATURES = [16, 32, 64]
WEIGHT_DECAY = [0.00005, 0.0005]
BS = 10


process Mean {
    executor 'local'
    clusterOptions = "-S /bin/bash"

    input:
    file py from MEANPY
    file toannotate from IMAGE_FOLD
    output:
    file "mean_file.npy" into MeanFile, MeanFile2

    """
    python $py --path $toannotate --output .
    """
}

process CreateTFRecords {
    clusterOptions = "-S /bin/bash -l mem_free=20G"
//    queue = "all.q"
//    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD

    output:
    file "UNet.tfrecords" into DATAQUEUE_TRAIN

    """
    PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --output UNet.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type Normal --train
    """
}

process Training {

    clusterOptions = "-S /bin/bash -q cuda.q"
    publishDir "./Training", mode: "copy", overwrite: false
    maxForks = 2

    input:
    file tfrecord from DATAQUEUE_TRAIN
    file path from IMAGE_FOLD
    file py from PY
    val bs from BS
    val home from params.home
    each feat from ARCH_FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile
    output:
    file "${feat}_${wd}_${lr}" into RESULTS, RESULTS2

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
    }
    pyglib $py --tf_record $tfrecord --epoch 80 --path $path --log . --learning_rate $lr --batch_size $bs --n_features $feat --weight_decay $wd --mean_file $_
    """
}


WSD = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

process Testing {
    clusterOptions = "-S /bin/bash -q all.q -l mem_free=20G"
    publishDir "./Testing", mode: "copy", overwrite: false
    


    input:
    file path from IMAGE_FOLD
    file py from PYTEST 
    file _ from MeanFile2 .last()
    each val from WSD
    file res from RESULTS
    output:
    file "${res}__${val}.csv" into RESULTS_TEST

    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --path $path --output ${res}__${val}.csv --log $res --batch_size 1 --n_features ${res.baseName.split('_')[0]} --mean_file $_ --lambda $val --thresh 0.5
    """
}

process RegroupResults {
    clusterOptions = "-S /bin/bash -q all.q"
    publishDir "./Bestmodels", mode: "copy", overwrite: false

    input:
    file _ from RESULTS_TEST .toList()
    file res from RESULTS2 .toList()
    output:
    file "best_model" into BEST_MODEL
    file "general.csv" into GENERAL

    script:
    """
    #!/usr/bin/env python

    from glob import glob
    import pandas as pd 
    from os.path import join
    from UsefulFunctions.RandomUtils import textparser
    from os.path import join
    import os 
    CSV = glob('*.csv')
    res = []
    for k, f in enumerate(CSV):
        tmp = pd.read_csv(f, index_col=0)
        f = '.'.join(f.split('.')[:-1])
        f, lamb = f.split('__')
        feat, lr, wd = f.split('_')
        tmp.ix[0, 'Features'] = feat
        tmp.ix[0, 'Learning rate'] = lr 
        tmp.ix[0, 'Weight decay'] = wd
        res.append(tmp)
    res = pd.concat(res)
    res = res.set_index(['Features', 'Learning rate', 'Weight decay','Lambda'])
    res.to_csv('general.csv')
    best_index = res['AJI'].argmax()
    TOMOVE = ["_".join(best_index[0:3])]
    for file in TOMOVE:
        os.mkdir('best_model')
        os.rename(file, os.path.join('best_model', file))
    """
}
