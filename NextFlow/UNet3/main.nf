#!/usr/bin/env nextflow
// nextflow main.nf -profile GPU_thal --epoch 1 --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter -resume
//params.image_dir = '/data/users/pnaylor/Bureau'
params.image_dir = "/share/data40T_v2/Peter/Data"
//params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.python_dir = "/share/data40T_v2/Peter/PythonScripts/PhD_Fabien"
//params.home = "/data/users/pnaylor"
params.home = "/share/data40T_v2/Peter"
params.epoch = 40


IMAGES = file(params.image_dir + "/ToAnnotate")

BINTO3 = file('src/BinTo3.py')
PY = file('src/Training.py')
TENSORBOARD = file(params.image_dir + '/tensorboard_withmean')
MEANPY = file(params.python_dir + '/Data/MeanCalculation.py')
TFRECORDS = file('src/TFRecords.py')
PYTEST = file('src/Testing.py')
LEARNING_RATE = [0.001, 0.0001, 0.00001]
ARCH_FEATURES = [16, 32, 64]
WEIGHT_DECAY = [0.00005, 0.0005]
BS = 10
DISKSIZE = [2, 3, 4]

process BinTo3 {
    clusterOptions "-S /bin/bash"
    input:
    file py from BINTO3
    file toannotate from IMAGES
    each disk from DISKSIZE
    output:
    file 'ToAnnotate3_*' into IMAGE_FOLD, IMAGE_FOLD2, IMAGE_FOLD3, IMAGE_FOLD4

    """
    python $py $disk
    """
}

process Mean {
    executor 'local'
    clusterOptions "-S /bin/bash"

    input:
    file py from MEANPY
    file toannotate from IMAGES
    output:
    file "mean_file.npy" into MeanFile, MeanFile2

    """
    python $py --path $toannotate --output .
    """
}


def Assign( file ) {
    disk = file.name.split('_')[1].toInteger()
    if( disk == 2 ) {
        15
    }
    else { 
        if( disk == 3 ) {
            16
        }
        else {
            17
        }
    }
}
def Assign2( file ) {
    disk = file.name.split('_')[1].toInteger()
    if( disk == 2 ) {
        24
    }
    else { 
        if( disk == 3 ) {
            19
        }
        else {
            20
        }
    }
}
IMAGE_FOLD  .map { file -> tuple(Assign(file), file) }
                 .set { IMAGE_FOLD_AND_COMP }

IMAGE_FOLD2  .map { file -> tuple(Assign2(file), file) }
                 .set { IMAGE_FOLD_AND_COMP2 }

process CreateTFRecords {
    maxForks 3
    clusterOptions  "-S /bin/bash -l mem_free=20G -q all.q@compute-0-${key}"
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    set key, file(path) from IMAGE_FOLD_AND_COMP
    output:
    file "UNet_${path.name.split('_')[1]}.tfrecords" into DATAQUEUE_TRAIN
    """
    function pyglib {
        PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --output UNet_${path.name.split('_')[1]}.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type JUST_READ --train $key
    """
}


process CreateTFRecords2 {
    maxForks 3
    clusterOptions  "-S /bin/bash -l mem_free=20G -q all.q@compute-0-${key}"
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    set key, file(path) from IMAGE_FOLD_AND_COMP2

    output:
    file "UNet_${path.name.split('_')[1]}.tfrecords" into DATAQUEUE_TRAIN2

    """
    function pyglib {
        PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --output UNet_${path.name.split('_')[1]}.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type JUST_READ --train
    """
}

// we have to phase the outputs of create and paths

def getPositionID( file ) {
        file.baseName.split('_')[1]
}
IMAGE_FOLD3 .phase(DATAQUEUE_TRAIN, remainder: true) { it -> getPositionID(it) } .set { PATH_AND_RECORD }
IMAGE_FOLD4 .phase(DATAQUEUE_TRAIN2, remainder: true) { it -> getPositionID(it) } .set { PATH_AND_RECORD2 }


process Training {

    clusterOptions = "-S /bin/bash -q cuda.q"
    publishDir "./Training", mode: "copy", overwrite: false
    maxForks = 2

    input:
    set file(path), file(tfrecord) from PATH_AND_RECORD
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
