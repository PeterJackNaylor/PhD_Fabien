
params.inputfolder = "/share/data40T_v2/CAMELYON16_data/"
TIFF_TUMOR = file(params.inputfolder + "/Tumor/*.tif")
TIFF_NORMAL  = file(params.inputfolder + "/Normal/*.tif")


ROI_PY = file('ChopPatient.py')
TRAINTESTPY = file("TrainTestSet.py")
VGG16 = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/NewStuff/vgg16.py')


ALL_FOLDER = file("/share/data40T_v2/CAMELYON16_precut")
TENSORBOARD = file('/share/data40T_v2/TrainingOutput')

TICKET_VAL = 60
SPLIT_RATIO = 0.1


process ChopNormalPatient {

    executor 'sge'
    profile = 'cluster'
    queue = "all.q"
    validExitStatus 0
    clusterOptions = "-S /bin/bash -l mem_free=1G"
    publishDir ALL_FOLDER, overwrite: false
    maxForks = 100
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file normal from TIFF_NORMAL
    file pythonfile from ROI_PY
    val ticket_val from TICKET_VAL

    output:
    file "${normal.getBaseName()}.txt" into DB_N
    file "*.png" into DB_N_IMG
    
    """
    python $ROI_PY --output . --type 'Normal' --file $normal --ticket_val $ticket_val
    """
}

process ChopTumorPatient {

    executor 'sge'
    profile = 'cluster'
    queue = "all.q"
    validExitStatus 0
    clusterOptions = "-S /bin/bash -l mem_free=1G"
    publishDir ALL_FOLDER, overwrite: false
    maxForks = 100
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file tumor from TIFF_TUMOR
    file pythonfile from ROI_PY

    output:
    file "${tumor.getBaseName()}.txt" into DB_T
    file "*.png" into DB_T_IMG

    """
    python $ROI_PY --output . --type 'Tumor' --file $tumor
    """
}



process CreateTrainTestSet {
    executor 'local'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash -l h_vmem=1G"
    publishDir ALL_FOLDER, mode: "copy", overwrite: false
    maxForks = 1

    input:
    val split_value from SPLIT_RATIO
    val txt_fold from ALL_FOLDER
    file traintestset from TRAINTESTPY
    file wait from DB_N .mix(DB_T) .toList()

    output:
    file "train.txt" into TRAIN_TXT
    file "test.txt" into TEST_TXT


    script:
    """
    python $traintestset --input $txt_fold --output . --split_value $split_value
    """
}


LEARNING_RATE = [0.001, 0.0001, 0.0001]
ARCH_FEATURES = [2, 4, 8, 16, 32, 64]
WEIGHT_DECAY = [0.0005, 0.00005]
BS = 128

process Training {

    executor 'sge'
    profile = 'GPU'
    validExitStatus 0 
    queue = "cuda.q"
    clusterOptions = "-S /bin/bash"
    publishDir TENSORBOARD, mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from ALL_FOLDER
    file vgg from VGG16
    val bs from BS
    each feat from ARCH_FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY
    file train_txt from TRAIN_TXT

    output:
    file "${feat}_${lr}" into RESULTS

    beforeScript "source /share/data40T_v2/Peter/CUDA_LOCK/.whichNODE"
    afterScript "source /share/data40T_v2/Peter/CUDA_LOCK/.freeNODE"

    script:
    """
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $vgg --epoch 5 --path $path --log . --learning_rate $lr --batch_size $bs --n_features $feat --weight_decay $wd

    """
}

