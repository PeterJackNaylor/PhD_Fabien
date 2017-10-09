#!/usr/bin/env nextflow

/// nextflow HEAD.nf -profile GPU --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter --cellcogn /share/data40T_v2/Peter/Data/CellCognition/ -resume

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "/data/users/pnaylor/Bureau/CellCognition"

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
PY = file(params.python_dir + '/Nets/UNetBatchNorm_v2.py')
TENSORBOARD = file(params.image_dir + '/tensorboard_DA')
MEANPY = file(params.python_dir + '/Data/MeanCalculation.py')
TFRECORDS = file(params.python_dir + '/Data/CreateTFRecords_DA.py')

LEARNING_RATE = [0.0001]
ARCH_FEATURES = [32]
WEIGHT_DECAY = [ 0.00005]
HE = [0.01, 0.07, 0.1, 0.15, 0.20, 0.25]
HSV = [0.01, 0.07, 0.1, 0.15, 0.20, 0.25]
ELAST1 = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
ELAST2 = [0.01, 0.02, 0.03, 0.04, 0.046875, 0.05, 0.06, 0.07]
ELAST3 = [0.01, 0.07, 0.1, 0.15, 0.20, 0.25]
BS = 10
params.epoch = 1 

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


process CreateTFRecords_he {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    each he1 from HE
    each he2 from HE

    output:
    file "HE_${he1}_${he2}.tfrecords" into DATAQUEUE_HE
    """

    python $py --output HE_${he1}_${he2}.tfrecords --he1 $he1 --he2 $he2 --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type Normal --train
    """
}

process CreateTFRecords_hsv {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    each hsv1 from HSV
    each hsv2 from HSV

    output:
    file "HSV_${hsv1}_${hsv2}.tfrecords" into DATAQUEUE_HSV
    """

    python $py --output HSV_${hsv1}_${hsv2}.tfrecords --hsv1 $hsv1 --hsv2 $hsv2 --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type Normal --train
    """
}

process CreateTFRecords_elast {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    each elast1 from ELAST1
    each elast2 from ELAST2
    each elast3 from ELAST3

    output:
    file "Elast_${elast1}_${elast2}_${elast3}.tfrecords" into DATAQUEUE_ELAST
    """

    python $py --output Elast_${elast1}_${elast2}_${elast3}.tfrecords --elast1 $elast1 --elast2 $elast2 --elast3 $elast3 --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type Normal --train
    """
}

DATA = DATAQUEUE_HE.concat(DATAQUEUE_HSV, DATAQUEUE_ELAST)

process Training {
//    scratch '/scratch'
    clusterOptions = "-S /bin/bash"
//   publishDir TENSORBOARD, mode: "copy", overwrite: false
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
    file __ from DATA
    val epoch from params.epoch
    output:
    file './${__.getBaseName().split("_")[0]}_*' into RESULTS 

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --tf_record $__ --path $path  --log ./${__.getBaseName().split('.tfrecord')[0]} --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100

    """
}


