#!/usr/bin/env nextflow

/// nextflow HEAD.nf -profile GPU --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter --cellcogn /share/data40T_v2/Peter/Data/CellCognition/ -resume

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "/data/users/pnaylor/Bureau/CellCognition"

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
PY = file(params.python_dir + '/Nets/UNetMultiClass_v2.py')
TENSORBOARD = file(params.image_dir + '/tensorboard_reduceclass')
MEANPY = file(params.python_dir + '/Data/MeanCalculation.py')
BinToColorPy = file(params.python_dir + '/PrepareData/XmlParsing.py')
SlideName = file(params.python_dir + '/PrepareData/EverythingExceptColor.py')
CELLCOG_classif = file(params.cellcogn + '/classifier_January2017')
CELLCOG_folder = file(params.cellcogn + '/Fabien')
TFRECORDS = file(params.python_dir + '/Data/CreateTFRecords.py')

LEARNING_RATE = [0.01, 0.001, 0.0001]
ARCH_FEATURES = [16, 32, 64]
WEIGHT_DECAY = [0.005, 0.0005, 0.00005]
BS = 16
params.epoch = 1 

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

process CreateTFRecords {
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

    python $py --output UNetRecords.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type ReducedClass
    """
}

process CreateTFRecords2 {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from ToAnnotateColor2

    output:
    file "UNetRecords.tfrecords" into DATAQUEUE2
    """

    python $py --output UNetRecords.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type ReducedClass
    """
}


process Training {

    clusterOptions = "-S /bin/bash"
//   publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from ToAnnotateColor2
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
    python $py --tf_record $__ --path $path  --log . --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100

    """
}


process Training2 {

    clusterOptions = "-S /bin/bash"
    publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from IMAGE_FOLD
    file py from PY
    file res from RESULTS
    val bs from BS
    val home from params.home
//    val pat from PATIENT
    file _ from MeanFile
    file __ from DATAQUEUE2
    val epoch from params.epoch
    output:
    file res into RESULTS2

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --tf_record $__ --path $path  --log . --learning_rate ${res.split("_")[3]} --batch_size $bs --epoch $epoch --n_features ${res.split("_")[1]} --weight_decay ${res.split("_")[2]} --mean_file $_ --n_threads 100 --num_labels 6

    """
}


