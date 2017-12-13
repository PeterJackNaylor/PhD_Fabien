#!/usr/bin/env nextflow

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "/data/users/pnaylor/Bureau/CellCognition"
params.epoch = 1 



IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
PY = file(params.python_dir + '/Nets/UNetBatchNorm_v2.py')
MEANPY = file(params.python_dir + '/Data/MeanCalculation.py')
TFRECORDS = file('src/CreateTFRecords_DA.py')
TESTPY=file("src/Testing.py")


LEARNING_RATE = [0.0001]
ARCH_FEATURES = [32]
WEIGHT_DECAY = [ 0.00005]
H = Channel.from(0.01, 0.1, 0.2, 0.3, 0.4)
E = Channel.from(0.01, 0.1, 0.2, 0.3, 0.4)
BS = 10
H .combine(E) .into{HE}
node = []
comp1 = Channel.from( 15..24 )
comp2 = Channel.from( 15..24 )
comp3 = Channel.from( 15..24 )
comp1 .concat(comp2) .concat(comp3) .set {COMP}

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
    queue "compute-0-${c}@all.q"
    memory '60G'
    maxForks = 2
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val h, e from HE
    val c from  COMP
    output:
    file "HE_${he1}_${he2}.tfrecords" into DATAQUEUE_HE
    """
    function pyglib {
        PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --output HE_${he1}_${he2}.tfrecords --he1 $he1 --he2 $he2 --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type Normal --train
    """
}



process Training {
    clusterOptions = "-S /bin/bash"
    maxForks = 2
    queue 'cuda.q'
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
    file "${__.getBaseName().split('.tfrecord')[0]}" into RESULTS 

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --tf_record $__ --path $path  --log ./${__.getBaseName().split('.tfrecord')[0]} --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100
    echo 'Done' >> ${__.getBaseName().split('.tfrecord')[0]}/readme.md
    """
}

process Testing {
    clusterOptions = "-S /bin/bash"
    publishDir "./Test/"
    maxForks = 2
    input:
    file path from IMAGE_FOLD
    file py from TESTPY
    file folder from RESULTS
    val home from params.home
    file _ from MeanFile2
    output:
    file "${folder}.txt" into RES

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    function pyglib {
        PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --path $path --output $folder -f $folder/32_0.00005_0.0001 --mean_file $_
    """

}


process RegroupResults {
    clusterOptions = "-S /bin/bash"
    publishDir "./Results", overwrite: true

    input:
    file fold from RES .toList()
    output:
    file "results.csv" into RES
    """
    echo IMPLEMENT
    """
}

