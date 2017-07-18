#!/usr/bin/env nextflow

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "data/users/pnaylor/Bureau/CellCognition"

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
PY = file(params.python_dir + '/NewStuff/UNetMultiClass.py')
TENSORBOARD = file(params.image_dir + '/tensorboard_multiclass')
MEANPY = file(params.python_dir + '/NewStuff/MeanCalculation.py')
BinToColorPy = file(params.python_dir + '/PrepareData/XmlParsing.py')
CELLCOG_classif = file(params.cellcogn + '/classifier_January2017')
CELLCOG_folder = file(params.cellcogn + '/Fabien')

LEARNING_RATE = [0.0001, 0.00001, 0.0000001]
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

process BinToColor {
    executor 'local'
    clusterOptions = "-S /bin/bash"

    input:
    file py from BinToColorPy
    file toannotate from IMAGE_FOLD
    file classifier from CELLCOG_classif
    file cellcog_folder from CELLCOG_folder
    output:
    file "./ToAnnotateColor" into ToAnnotateColor

    """
    python XmlParsing.py --a $classifier --c $cellcog_folder --o ./ToAnnotateColor/
    """
}

process Training {

    clusterOptions = "-S /bin/bash"
    publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from ToAnnotateColor
    file py from PY
    val bs from BS
    val home from params.home
//    val pat from PATIENT
    each feat from ARCH_FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile
    output:
    file "${feat}_${wd}_${lr}" into RESULTS

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --epoch 500 --path $path --log . --learning_rate $lr --batch_size $bs --n_features $feat --weight_decay $wd

    """
}
