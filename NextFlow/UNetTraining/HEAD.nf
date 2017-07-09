#!/usr/bin/env nextflow

IMAGE_FOLD = file('/data/users/pnaylor/Bureau/ToAnnotate')
PY = file('/data/users/pnaylor/Documents/Python/PhD_Fabien/NewStuff/UNetBatchNorm.py')
TENSORBOARD = file('/data/users/pnaylor/Bureau/tensorboard')
MEANPY = file('/data/users/pnaylor/Documents/Python/PhD_Fabien/NewStuff/MeanCalculation.py')

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
    python $py --epoch 250 --path $path --log . --learning_rate $lr --batch_size $bs --n_features $feat --weight_decay $wd

    """
}
