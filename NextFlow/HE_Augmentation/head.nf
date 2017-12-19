#!/usr/bin/env nextfl◊ow

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
H .combine(E) .set{HE}
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
    file "mean_file.npy" into MeanFile, MeanFile2, MeanFileD

    """
    python $py --path $toannotate --output .
    """
}

process CreateTFRecords_he {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue "all.q@compute-0-${c}"
    memory '60G'
    maxForks = 8
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    set h, e from HE
    val c from  COMP
    output:
    file "HE_${h}_${e}.tfrecords" into DATAQUEUE_HE
    """
    function pyglib {
        PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py --output HE_${h}_${e}.tfrecords --he1 $h --he2 $e --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type Normal --train
    """
}

REPEAT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


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
    each rep from REPEAT    
    file _ from MeanFile
    file __ from DATAQUEUE_HE
    val epoch from params.epoch
    output:
    file "${__.getBaseName().split('.tfrecord')[0]}_${rep}" into RESULTS, RESULTS_DARKER 

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
    }
    pyglib $py --tf_record $__ --path $path  --log ./${__.getBaseName().split('.tfrecord')[0]}_${rep} --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100
    """
}

process Testing {
    publishDir "./Test/"
    clusterOptions "-S /bin/bash -l h_vmem=60G"
    memory '60G'
    input:
    file path from IMAGE_FOLD
    file py from TESTPY
    file folder from RESULTS
    val home from params.home
    file _ from MeanFile2
    output:
    file "${folder}.txt" into RES

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
    file "results.csv" into RES_csv
    """
#!/usr/bin/env python
from glob import glob
import pandas as pd
from os.path import join, basename
from UsefulFunctions.RandomUtils import textparser
filess = glob('*.txt')
import pdb
result = pd.DataFrame(columns=['Model', 'Param1', 'Param2', 'Repeat', 'AJI', 'Mean acc', 'Precision', 'Recall', 'F1', 'ACC'])

def name_parse(string):
    string = basename(string).split('.tx')[0]
    model = string.split('_')[0]
    Param1 = string.split('_')[1]
    Param2 = string.split('_')[2]
    Param3 = string.split('_')[3]
    return model, Param1, Param2, Param3

for k, f in enumerate(filess):
    model, p1, p2, p3 = name_parse(f)
    dic = textparser(f)
    dic['Param1'] = p1
    dic['Param2'] = p2
    dic['Repeat'] = p3
    dic['Model'] = model
    result.loc[k] = pd.Series(dic)
result = result.set_index(['Model', 'Param1', 'Param2', 'Repeat'])
result.to_csv('results.csv')
    """
}

TESTPYDARK = file('src/TestingDarker.py')
IMAGE_DARKER = file(params.image_dir + "/NeerajKumar/ForDataGenTrainTestVal")
process TestingDarker {

    publishDir "./TestDarker"
    clusterOptions "-S /bin/bash -l h_vmem=60G"
    memory '60G'
    input:
    file path from IMAGE_DARKER
    file py from TESTPYDARK
    file folder from RESULTS_DARKER
    val home from params.home
    file _ from MeanFileD
    output:
    file "${folder}.txt" into RES_DARKER

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
    publishDir "./ResultsDarker", overwrite: true

    input:
    file fold from RES_DARKER .toList()
    output:
    file "results.csv" into RES_DARKcsv
    """
#!/usr/bin/env python
from glob import glob
import pandas as pd
from os.path import join, basename
from UsefulFunctions.RandomUtils import textparser
filess = glob('*.txt')
import pdb
result = pd.DataFrame(columns=['Model', 'Param1', 'Param2', 'Repeat', 'AJI', 'Mean acc', 'Precision', 'Recall', 'F1', 'ACC'])

def name_parse(string):
    string = basename(string).split('.tx')[0]
    model = string.split('_')[0]
    Param1 = string.split('_')[1]
    Param2 = string.split('_')[2]
    Param3 = string.split('_')[3]
    return model, Param1, Param2, Param3

for k, f in enumerate(filess):
    model, p1, p2, p3 = name_parse(f)
    dic = textparser(f)
    dic['Param1'] = p1
    dic['Param2'] = p2
    dic['Repeat'] = p3
    dic['Model'] = model
    result.loc[k] = pd.Series(dic)
result = result.set_index(['Model', 'Param1', 'Param2', 'Repeat'])
result.to_csv('results.csv')
    """
}
