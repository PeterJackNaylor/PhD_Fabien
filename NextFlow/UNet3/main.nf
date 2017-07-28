#!/usr/bin/env nextflow
// nextflow main.nf -profile GPU_thal --epoch 1 --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter -resume
params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
PY = file(params.python_dir + '/NewStuff/UNet3_v2.py')
TENSORBOARD = file(params.image_dir + '/new_queue_3')
MEANPY = file(params.python_dir + '/NewStuff/MeanCalculation.py')
TFRECORDS = file(params.python_dir + '/NewStuff/CreateTFRecords.py')

LEARNING_RATE = [0.01, 0.001, 0.0001]
ARCH_FEATURES = [2, 4, 8, 16, 32]
WEIGHT_DECAY = [0.0005, 0.00005]
BS = 32

params.epoch = 1



process CreateTFRecords {
	clusterOptions = "-S /bin/bash"

	input:
	file py from TFRECORDS
	val epoch from params.epoch
    file path from IMAGE_FOLD

    output:
    file "UNetRecords.tfrecords" into DATAQUEUE


    """
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $py --output UNetRecords.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch
    """
}

process Mean {
    executor 'local'
    clusterOptions = "-S /bin/bash"

    input:
    file py from MEANPY
    file toannotate from IMAGE_FOLD
    output:
    file "mean_file.npy" into MeanFile

    """
    python $py --path $toannotate --output .
    """
}

process Training {

    clusterOptions = "-S /bin/bash"
    publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from IMAGE_FOLD
    file py from PY
    val bs from BS
    val home from params.home
//    val pat from PATIENT
    each feat from ARCH_FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile
    file __ from DATAQUEUE
    val epoch from params.epoch
    output:
    file "${feat}_${wd}_${lr}" into RESULTS

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $py --tf_record UNetRecords.tfrecords --path $path  --log . --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100

    """
}
