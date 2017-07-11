#!/usr/bin/env nextflow

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
PY = file(params.python_dir + '/NewStuff/UNetBatchNorm.py')
TENSORBOARD = file(params.image_dir + '/tensorboard_withmean')
MEANPY = file(params.python_dir + '/NewStuff/MeanCalculation.py')

LEARNING_RATE = [0.00001, 0.000001, 0.00000001]
ARCH_FEATURES = [2, 4, 8, 16, 32]
WEIGHT_DECAY = [0.0005, 0.00005]
BS = 32

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

    executor 'local'
    profile = 'GPU'
    validExitStatus 0 
    queue = "cuda.q"
    clusterOptions = "-S /bin/bash"
    publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from IMAGE_FOLD
    file py from PY
    val bs from BS
//    val pat from PATIENT
    each feat from ARCH_FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile
    output:
    file "${feat}_${wd}_${lr}" into RESULTS

    beforeScript "source /data/users/pnaylor/CUDA_LOCK/.whichNODE"
    afterScript "source /data/users/pnaylor/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --epoch 500 --path $path --log . --learning_rate $lr --batch_size $bs --n_features $feat --weight_decay $wd

    """
}
