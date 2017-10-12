

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "/data/users/pnaylor/Bureau/CellCognition"


LEARNING_RATE = [0.01, 0.001, 0.0001, 0.00001]
ARCH_FEATURES = [16, 32, 64]
WEIGHT_DECAY = [0.0005, 0.00005]
BS = 10
params.epoch = 1 
IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
MEANPY = file(params.python_dir + '/Data/MeanCalculation.py')
BinToDistanceFile = file('BinToDistance.py')
TFRECORDS = file(params.python_dir + '/Data/CreateTFRecords.py')
PY = file(params.python_dir + '/Nets/UNetDistance.py')
PY_PRETRAIN = file(params.python_dir + '/Nets/UNetDistancePretrain.py')
PRETRAINED = 

process Mean {
    executor 'local'
    clusterOptions = "-S /bin/bash"

    input:
    file py from MEANPY
    file toannotate from IMAGE_FOLD
    output:
    file "mean_file.npy" into MeanFile, MeanFile2, MeanFile3

    """
    python $py --path $toannotate --output .
    """
}


process BinToDistance {
    queue = "all.q"
    clusterOptions = "-S /bin/bash"
    input:
    file py from BinToDistanceFile
    file toannotate from IMAGE_FOLD
    output:
    file 'ToAnnotateDistance' into DISTANCE_FOLD, DISTANCE_FOLD2, DISTANCE_FOLD3, DISTANCE_FOLD4, DISTANCE_FOLD5, DISTANCE_FOLD6

    """
    python $py $toannotate
    """
}


process CreateTFRecords {
    clusterOptions = "-S /bin/bash -l mem_free=20G"
//    queue = "all.q"
//    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from DISTANCE_FOLD

    output:
    file "DistanceTrainRecords.tfrecords" into DATAQUEUE_TRAIN, DATAQUEUE_TRAIN1_2

    """
    PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python $py --output DistanceTrainRecords.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type JUST_READ --train
    """
}



process CreateTFRecords2 {
    clusterOptions = "-S /bin/bash -l h_vmem=20G"
//    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from DISTANCE_FOLD3

    output:
    file "DistanceTrainRecords.tfrecords" into DATAQUEUE_TRAIN2

    """
    PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python $py --output DistanceTrainRecords.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type JUST_READ --train
    """
}

process CreateTestTFRecords {
    clusterOptions = "-S /bin/bash -l mem_free=20G"
//    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from DISTANCE_FOLD2

    output:
    file "DistanceTestRecords.tfrecords" into DATAQUEUE_TEST

    """
    PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    alias pyglib='/share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python'
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python $py --output DistanceTestRecords.tfrecords --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type JUST_READ --test
    """
}


process Training {

    clusterOptions = "-S /bin/bash -q cuda.q"
//    publishDir "./FirstResult", mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from DISTANCE_FOLD4
    file py from PY
    val bs from BS
    val home from params.home
    each feat from ARCH_FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile
    file __ from DATAQUEUE_TRAIN
    val epoch from params.epoch
    output:
    file "${feat}_${wd}_${lr}" into RESULTS 

    beforeScript "source ${home}/CUDA_LOCK/.whichNODE"
    afterScript "source ${home}/CUDA_LOCK/.freeNODE"

    script:
    """
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $py --tf_record $__ --path $path  --log . --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100

    """
}


process Training2 {

    clusterOptions = "-S /bin/bash -q cuda.q"
   publishDir "./Results", mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from DISTANCE_FOLD5
    file py from PY
    val bs from BS
    file res from RESULTS
    val home from params.home
//    val pat from PATIENT
    file _ from MeanFile2
    file __ from DATAQUEUE_TRAIN2
    val epoch from params.epoch
    output:
    file res into RESULTS2

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $py --tf_record $__ --path $path  --log . --learning_rate ${res.name.split('_')[2]} --batch_size $bs --epoch $epoch --n_features ${res.name.split('_')[0]} --weight_decay ${res.name.split('_')[1]} --mean_file $_ --n_threads 100

    """
}


process PreTraining {

    clusterOptions = "-S /bin/bash -q cuda.q"
//    publishDir "./FirstResult", mode: "copy", overwrite: false
    maxForks = 2

    input:
    file path from DISTANCE_FOLD6
    file py from PY_PRETRAIN
    val bs from BS
    val home from params.home
    each feat from ARCH_FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile3
    file __ from DATAQUEUE_TRAIN1_2
    val epoch from params.epoch
    file pretrain from PRETRAINED
    output:
    file "${feat}_${wd}_${lr}" into RESULTS 

    beforeScript "source ${home}/CUDA_LOCK/.whichNODE"
    afterScript "source ${home}/CUDA_LOCK/.freeNODE"

    script:
    """
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $py --tf_record $__ --path $path  --log . --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100 --pretrainmodel $pretrain

    """
}