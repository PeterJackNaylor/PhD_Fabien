#!/usr/bin/env nextflow

/// nextflow HEAD.nf -profile GPU --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter --cellcogn /share/data40T_v2/Peter/Data/CellCognition/ -resume

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
params.cellcogn = "/data/users/pnaylor/Bureau/CellCognition"
params.epoch = 40
EPOCHUNET = 80

TENSORBOARDUNET = file(params.image_dir + '/tensorboard_unet')
TENSORBOARDDIST = file(params.image_dir + '/tensorboard_dist')

TENSORBOARD_BIN_32 = file(params.image_dir + '/tensorboard_fcn_bin_32')
TENSORBOARD_BIN_16 = file(params.image_dir + '/tensorboard_fcn_bin_16')
TENSORBOARD_BIN_8 = file(params.image_dir + '/tensorboard_fcn_bin_8')
TENSORBOARD_BIN_8_NEE = file(params.image_dir + '/tensorboard_fcn_bin_8_no_cross')
RESULTS_DIR = file(params.image_dir + '/results')

TFRECORDS = file('src/TFRecords.py')
FCN32TRAIN = file("src/FCN32Train.py")
FCN32TEST = file('src/FCN32Test.py')
FCN16TRAIN = file("src/FCN16Train.py")
FCN16TEST = file('src/FCN16Test.py')
FCN8TRAIN = file("src/FCN8Train.py")
FCN8TEST = file('src/FCN8Test.py')
FCN8VAL = file('src/FCN8Val.py')
TFRECORDS_VAL = file('src/TFRecordsVal.py')
UNET = file('src/UNetTraining.py')
UNNETTEST = file('src/UNetTesting.py')
UNNETVAL = file('src/UNetVal.py')
PLOT_RES = file('src/plot.py')

MEANPY = file('src/MeanCalculation.py')
BinToColorPy = file(params.python_dir + '/PrepareData/XmlParsing.py')
SlideName = file(params.python_dir + '/PrepareData/EverythingExceptColor.py')
GivingBackIdea = file("GivingBackID.py")
CELLCOG_classif = file(params.cellcogn + '/classifier_January2017')
CELLCOG_folder = file(params.cellcogn + '/Fabien')

IMAGE_FOLD = file(params.image_dir + "/ForDataGenTrainTestVal")
CHECKPOINT_VGG = file(params.image_dir + "/pretrained/vgg_16.ckpt")
IMAGE_SIZE = 224
IMAGE_SIZEUNET = 212
IMAGE_SIZE_TEST = 500
IMAGE_SIZE_VAL = 1000
NPRINT = 100
ITERTEST = 24

ITER32 = 10800
ITER16 = 10800
ITER8 = 10800

LEARNING_RATE_32 = [0.001, 0.0001, 0.00001, 0.000001]
LEARNING_RATE_16 = [0.0001, 0.00001, 0.000001, 0.0000001]
LEARNING_RATE_8 = [0.00001, 0.000001, 0.0000001, 0.00000001]
LEARNING_RATE_UNET = [0.001, 0.0001, 0.00001]

ARCH_FEATURES_UNET = [16, 32, 64]
WEIGHT_DECAY_UNET = [0.00005, 0.0005]
BS_UNET = 10

process CreateTrainRecords {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZE

    output:
    file "TrainBin.tfrecords" into TrainRECORDBIN
    """
    python $py --output TrainBin.tfrecords --path $path --crop 16 --no-UNet --size $i_s --seed 42 --epoch $epoch --type JUST_READ --split train
    """
}

process CreateTrainRecordsUnet {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZEUNET

    output:
    file "TrainBin.tfrecords" into TrainRECORDBINUNET
    """
    python $py --output TrainBin.tfrecords --path $path --crop 16 --UNet --size $i_s --seed 42 --epoch $epoch --type JUST_READ --split train
    """
}

process CreateTestRecords {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZE_TEST

    output:
    file "TestBin.tfrecords" into TestRECORDBIN
    """
    python $py --output TestBin.tfrecords --path $path --crop 1 --no-UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split test
    """
}

process CreateTestRecordsUnet {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZEUNET

    output:
    file "TestBin.tfrecords" into TestRECORDBINUNET
    """
    python $py --output TestBin.tfrecords --path $path --crop 4 --UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split test
    """
}

process CreateValidationRecords {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZE_VAL

    output:
    file "ValBin.tfrecords" into ValRECORDBIN
    """
    python $py --output ValBin.tfrecords --path $path --crop 1 --no-UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split validation
    """
}

process CreateValidationRecordsUnet {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZEUNET

    output:
    file "ValBin.tfrecords" into ValRECORDBINUNET, ValRECORDBINUNET_NEE
    """
    python $py --output ValBin.tfrecords --path $path --crop 16 --UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split validation
    """
}

process Mean {
    executor 'local'
    clusterOptions = "-S /bin/bash"

    input:
    file py from MEANPY
    file toannotate from IMAGE_FOLD
    output:
    file "mean_file.npy" into MeanFile, MeanFile2, MeanFile3, MeanFile4, MeanFile5, MeanFile6, MeanFile7, MeanFile4_NEE, MeanFile7_NEE

    """
    python $py --path $toannotate --output .
    """
}

process UNetTraining {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARDUNET, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from IMAGE_FOLD
    file py from UNET
    val bs from BS_UNET
    val home from params.home
    each feat from ARCH_FEATURES_UNET
    each lr from LEARNING_RATE_UNET
    each wd from WEIGHT_DECAY_UNET    
    file _ from MeanFile
    file __ from TrainRECORDBINUNET
    val epoch from EPOCHUNET
    output:
    file "${feat}_${wd}_${lr}" into RES_UNET, RES_UNET2, RES_UNET_NEE

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --tf_record $__ --path $path  --log . --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100
    """
}


LAMBDA = [7]
process UNetTest {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARDUNET, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from IMAGE_FOLD
    file py from UNNETTEST
    val bs from BS_UNET
    val home from params.home
    file _ from MeanFile3
    file __ from TestRECORDBINUNET
    each lamb from LAMBDA
    file res from RES_UNET
    output:
    file "${res.name}_${lamb}.csv" into UNET_TEST

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --path $path  --log ${res} --batch_size $bs --n_features ${res.name.split('_')[0]} --mean_file $_ --lambda $lamb --thresh 0.5 --output ${res.name}_${lamb}.csv
    """
}

process RegroupUNET_results {
    publishDir TENSORBOARDUNET, mode: "copy", overwrite: true

    input:
    file _ from UNET_TEST .toList()
    file __ from RES_UNET2 .toList()
    output:
    file "bestmodel_*" into BEST_UNET
    file "UNET_results.csv" into UNET_table
//    file "log__fcn32__*" into PUBLISH32
    """
    #!/usr/bin/env python
    import os
    import pandas as pd
    import pdb
    from glob import glob
    CSV = glob('*.csv')
    df_list = []
    for f in CSV:
        df = pd.read_csv(f, index_col=False)
        df.index = ["_".join(f.split('.')[:-1])]
        df_list.append(df)
    table = pd.concat(df_list)
    best_index = table[' F1'].argmax()
    table.to_csv('UNET_results.csv')
    tmove_name = "{}".format(best_index).split('_')[:-1]
    res = ""
    for el in tmove_name:
        res += el
        if el == "0":
            res += "."
        else:
            res += "_"
    res = res[:-1]
    TOMOVE = glob(res)
    n_feat = res.split('_')[0]
    name = 'bestmodel_' + n_feat
    os.mkdir(name)
    for file in TOMOVE:
        os.rename(file, os.path.join(name, file))
    """
}

process UNetVal {

    clusterOptions = "-S /bin/bash"
    publishDir TENSORBOARDUNET, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from IMAGE_FOLD
    file py from UNNETVAL
    val home from params.home
    file _ from MeanFile4
    file __ from ValRECORDBINUNET
    val lamb from LAMBDA
    file res from BEST_UNET
    output:
    file "UNET_VAL.csv" into UNET_VAL

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --path $path  --log ${res} --batch_size 1 --n_features ${res.name.split('_')[1]} --mean_file $_ --lambda $lamb --thresh 0.5 --output UNET_VAL.csv
    """
}

process FCN32 {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARD_BIN_32, overwrite: true, pattern: "log__*"
    maxForks = 2

    input:
    file py from FCN32TRAIN
    val home from params.home
    file cp from CHECKPOINT_VGG
    file tfr from TrainRECORDBIN
    val i_s from IMAGE_SIZE
    each lr from LEARNING_RATE_32
    val np from NPRINT
    val iter from ITER32
    output:
    file "log__fcn32__*" into RESULTS_32
    file "model__*__fcn32s.ckpt.*" into CHECKPOINT_32
    file "model__*__fcn32s.ckpt.*" into CHECKPOINT_32_1 mode flatten
    
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --checkpoint $cp --checksavedir . --tf_records $tfr --log . --image_size $i_s --labels 2 --lr $lr --n_print $np --iter $iter
    """
}

process FCN32_testing {
    input:
    file py from FCN32TEST
    file cp from CHECKPOINT_32
    file tfr from TestRECORDBIN
    val iter from ITERTEST
    val home from params.home
    output:
    file "*.csv" into RES32

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records $tfr --labels 2 --iter $iter
    """

}

process RegroupFCN32_results {

    publishDir TENSORBOARD_BIN_32, mode: "copy", overwrite: true

    input:
    file _ from RES32 .toList()
    file __ from CHECKPOINT_32_1 .toList()
    file ___ from RESULTS_32 .toList()
    output:
    file "bestmodel" into STARTING_16
    file "FCN32_results.csv" into RES32_table

//    file "log__fcn32__*" into PUBLISH32
    """
    #!/usr/bin/env python
    import os
    os.mkdir('bestmodel')
    import pandas as pd
    import pdb
    from glob import glob
    CSV = glob('*.csv')
    df_list = []
    for f in CSV:
        df = pd.read_csv(f, index_col=0)
        df.index = [f[0:-4].split('__')[1]]
        df_list.append(df)
    table = pd.concat(df_list)

    best_index = table['F1'].argmax()
    table.to_csv('FCN32_results.csv')
    
    TOMOVE = glob("model__{}__fcn32s.ckpt*".format(best_index))
    for file in TOMOVE:
        os.rename(file, os.path.join("bestmodel", file))
    """

}

process FCN16 {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARD_BIN_16, overwrite: true, pattern: "log__*"
    maxForks = 2

    input:
    file py from FCN16TRAIN
    val home from params.home
    file cp from STARTING_16
    file tfr from TrainRECORDBIN
    val i_s from IMAGE_SIZE
    each lr from LEARNING_RATE_16
    val np from NPRINT
    val iter from ITER16
    output:
    file "log__fcn16__*" into RESULTS_16
    file "model__*__fcn16s.ckpt.*" into CHECKPOINT_16
    file "model__*__fcn16s.ckpt.*" into CHECKPOINT_16_1 mode flatten
    
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --checkpoint $cp --checksavedir . --tf_records $tfr --log . --image_size $i_s --labels 2 --lr $lr --n_print $np --iter $iter
    """
}

process FCN16_testing {
    input:
    file py from FCN16TEST
    file cp from CHECKPOINT_16
    file tfr from TestRECORDBIN
    val iter from ITERTEST
    val home from params.home
    output:
    file "*.csv" into RES16

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records $tfr --labels 2 --iter $iter
    """
}

process RegroupFCN16_results {

    publishDir TENSORBOARD_BIN_16, mode: "copy", overwrite: true

    input:
    file _ from RES16 .toList()
    file __ from CHECKPOINT_16_1 .toList()
    file ___ from RESULTS_16 .toList()

    output:
    file "bestmodel" into STARTING_8
    file "FCN16_results.csv" into RES16_table
//    file "log__fcn16__*" into PUBLISH16

    """
    #!/usr/bin/env python
    import os
    os.mkdir('bestmodel')
    import pandas as pd
    import pdb
    from glob import glob
    CSV = glob('*.csv')
    df_list = []
    for f in CSV:
        df = pd.read_csv(f, index_col=0)
        df.index = [f[0:-4].split('__')[1]]
        df_list.append(df)
    table = pd.concat(df_list)

    best_index = table['F1'].argmax()
    table.to_csv('FCN16_results.csv')
    
    TOMOVE = glob("model__{}__fcn16s.ckpt*".format(best_index))
    for file in TOMOVE:
        os.rename(file, os.path.join("bestmodel", file))
    """
}

process FCN8 {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARD_BIN_8, overwrite: true, pattern: "log__*"
    maxForks = 2

    input:
    file py from FCN8TRAIN
    val home from params.home
    file cp from STARTING_8
    file tfr from TrainRECORDBIN
    val i_s from IMAGE_SIZE
    each lr from LEARNING_RATE_8
    val np from NPRINT
    val iter from ITER8
    output:
    file "log__fcn8__*" into RESULTS_8
    file "model__*__fcn8s.ckpt.*" into CHECKPOINT_8, CHECKPOINT_8_NEE
    file "model__*__fcn8s.ckpt.*" into CHECKPOINT_8_1 mode flatten
    
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --checkpoint $cp --checksavedir . --tf_records $tfr --log . --image_size $i_s --labels 2 --lr $lr --n_print $np --iter $iter
    """
}

process FCN8_testing {
    input:
    file py from FCN8TEST
    file cp from CHECKPOINT_8
    file tfr from TestRECORDBIN
    val iter from ITERTEST
    val home from params.home
    output:
    file "*.csv" into RES8

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records $tfr --labels 2 --iter $iter
    """
}

process RegroupFCN8_results {

    publishDir TENSORBOARD_BIN_8, mode: "copy", overwrite: true


    input:
    file _ from RES8 .toList()
    file __ from CHECKPOINT_8_1 .toList()
    file ___ from RESULTS_8 .toList()

    output:
    file "bestmodel" into BEST_FCN8
    file "FCN8_results.csv" into RES8_table
//    file "log__fcn8__*" into PUBLISH8

    """
    #!/usr/bin/env python
    import os
    os.mkdir('bestmodel')
    import pandas as pd
    import pdb
    from glob import glob
    CSV = glob('*.csv')
    df_list = []
    for f in CSV:
        df = pd.read_csv(f, index_col=0)
        df.index = [f[0:-4].split('__')[1]]
        df_list.append(df)
    table = pd.concat(df_list)

    best_index = table['F1'].argmax()
    table.to_csv('FCN8_results.csv')
    
    TOMOVE = glob("model__{}__fcn8s.ckpt*".format(best_index))
    for file in TOMOVE:
        os.rename(file, os.path.join("bestmodel", file))
    """
}

ORGANS = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach"]

process FCN8_val_org {
    publishDir TENSORBOARD_BIN_8, mode: "copy", overwrite: true
    input:
    file pre_py from TFRECORDS_VAL
    file py from FCN8VAL
    file cp from BEST_FCN8
    val home from params.home
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZE_VAL
    each organ from ORGANS
    output:
    file "FCN_${organ}.csv" into FCN_VAL_RES

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    python $pre_py --output ${organ}.tfrecords --path $path --crop 1 --no-UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split validation --organ $organ 
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records ${organ}.tfrecords --labels 2 --iter 2 --output FCN_${organ}.csv
    """
}

// Distance method part
BinToDistanceFile = file('src/BinToDistance.py')
DISTTFRECORDS = file('src/DistTFRecords.py')
process BinToDistance {
    queue = "all.q"
    clusterOptions = "-S /bin/bash"
    input:
    file py from BinToDistanceFile
    file toannotate from IMAGE_FOLD
    output:
    file 'ToAnnotateDistance' into DISTANCE_FOLD, DISTANCE_FOLD2, DISTANCE_FOLD3, DISTANCE_FOLD4, DISTANCE_FOLD5, DISTANCE_FOLD6, DISTANCE_FOLD6_NEE

    """
    python $py $toannotate
    """
}

process CreateTrainRecordsDist {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from DISTTFRECORDS
    val epoch from params.epoch
    file path from DISTANCE_FOLD
    val i_s from IMAGE_SIZEUNET

    output:
    file "TrainDistance.tfrecords" into TrainRECORDDISTUNET
    """
    python $py --output TrainDistance.tfrecords --path $path --crop 16 --UNet --size $i_s --seed 42 --epoch $epoch --type JUST_READ --split train
    """
}

process CreateTestRecordsDist {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from DISTTFRECORDS
    val epoch from params.epoch
    file path from DISTANCE_FOLD2
    val i_s from IMAGE_SIZEUNET

    output:
    file "TestDistance.tfrecords" into TestRECORDDISTUNET
    """
    python $py --output TestDistance.tfrecords --path $path --crop 4 --UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split test
    """
}

process CreateValidationRecordsDist {
    clusterOptions = "-S /bin/bash -l h_vmem=60G"
    queue = "all.q"
    memory = '60G'
    input:
    file py from DISTTFRECORDS
    val epoch from params.epoch
    file path from DISTANCE_FOLD3
    val i_s from IMAGE_SIZEUNET

    output:
    file "ValDistance.tfrecords" into ValRECORDDISTUNET, ValRECORDDISTUNET_NEE
    """
    python $py --output ValDistance.tfrecords --path $path --crop 16 --UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split validation
    """
}

PYDIST = file('src/UNetDistanceTraining.py')
DISTTEST = file('src/UNetDistanceTesting.py')
DISTVAL = file('src/UNetDistanceVal.py')

P1= [1, 2, 3]

process UNetDistTraining {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARDDIST, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from DISTANCE_FOLD4
    file py from PYDIST
    val bs from BS_UNET
    val home from params.home
    each feat from ARCH_FEATURES_UNET
    each lr from LEARNING_RATE_UNET
    each wd from WEIGHT_DECAY_UNET    
    file _ from MeanFile5
    file __ from TrainRECORDDISTUNET
    val epoch from EPOCHUNET
    output:
    file "${feat}_${wd}_${lr}__Dist" into RES_DIST, RES_DIST2, RES_DIST_NEE

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --tf_record $__ --path $path  --log . --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100
    """
}
process UNetDistTest {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARDDIST, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from DISTANCE_FOLD5
    file py from DISTTEST
    val bs from BS_UNET
    val home from params.home
    file _ from MeanFile6
    file __ from TestRECORDDISTUNET
    each p from P1
    file res from RES_DIST
    output:
    file "${res.name}_${p}.csv" into DIST_TEST

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --path $path  --log ${res} --batch_size $bs --n_features ${res.name.split('_')[0]} --mean_file $_ --lambda $p --thresh 0.9 --output ${res.name}_${p}.csv
    """
}
process RegroupDIST_results {
    publishDir TENSORBOARDDIST, mode: "copy", overwrite: true

    input:
    file _ from DIST_TEST .toList()
    file __ from RES_DIST2 .toList()
    output:
    file "bestmodel_*" into BEST_Dist
    file "Dist_results.csv" into Dist_table
//    file "log__fcn32__*" into PUBLISH32
    """
    #!/usr/bin/env python
    import os
    import pandas as pd
    import pdb
    from glob import glob
    CSV = glob('*.csv')
    df_list = []
    for f in CSV:
        df = pd.read_csv(f, index_col=False)
        name = f.split('__')[0].split('/')[-1]
        df.index = [name]
        df_list.append(df)
    table = pd.concat(df_list)
    best_index = table[' F1'].argmax()
    table.to_csv('Dist_results.csv')
    tmove_name = "{}".format(best_index)
    TOMOVE = glob(tmove_name + "__Dist")
    n_feat = tmove_name.split('_')[0]
    name = 'bestmodel_' + n_feat
    os.mkdir(name)
    for file in TOMOVE:
        os.rename(file, os.path.join(name, file))
    """
}

process DistVal {

    clusterOptions = "-S /bin/bash"
    publishDir TENSORBOARDDIST, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from DISTANCE_FOLD6
    file py from DISTVAL
    val home from params.home
    file _ from MeanFile7
    file __ from ValRECORDDISTUNET
    file res from BEST_Dist
    file test_res from Dist_table
    output:
    file "DIST_VAL.csv" into DIST_VAL

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --path $path  --log ${res} --batch_size 1 --n_features ${res.name.split('_')[1]} --mean_file $_ --thresh 0.9 --output DIST_VAL.csv --test_res $test_res
    """
}

NEERAJ = file('Neeraj_PAPER.csv')

process RegroupPlot {
    publishDir './Final/', mode: 'copy', overwrite: true

    input:
    file _ from UNET_VAL .collect()
    file __ from FCN_VAL_RES .collect()
    file ___ from DIST_VAL .collect()
    file py from PLOT_RES
    file ___ from NEERAJ
    output:
    file "VSPaper.csv"
    file "*.png"
    """
    python $py
    """
}


// Here we will not be cross validating on my dataset




process FCN8_val_NEE {
//    publishDir TENSORBOARD_BIN_8_NEE, mode: "copy", overwrite: true
    input:
    file pre_py from TFRECORDS_VAL
    file py from FCN8VAL
    file cp from CHECKPOINT_8_NEE
    val home from params.home
    file path from IMAGE_FOLD
    val i_s from IMAGE_SIZE_VAL
    each organ from ORGANS
    output:
    file "FCN_${organ}.csv" into FCN_VAL_RES_NEE

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    """
    python $pre_py --output ${organ}.tfrecords --path $path --crop 1 --no-UNet --size $i_s --seed 42 --epoch 1 --type JUST_READ --split validation --organ $organ 
    python $py --checkpoint ${cp.first().name.split('.data')[0]} --tf_records ${organ}.tfrecords --labels 2 --iter 2 --output FCN_${organ}.csv
    """
}

process UNetVal_NEE {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARDUNET_NEE, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from IMAGE_FOLD
    file py from UNNETVAL
    val home from params.home
    file _ from MeanFile4_NEE
    file __ from ValRECORDBINUNET_NEE
    val lamb from LAMBDA
    file res from RES_UNET_NEE
    output:
    file "${res.name}_${lamb}.csv" into UNET_VAL_NEE

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --path $path  --log ${res} --batch_size 1 --n_features ${res.name.split('_')[0]} --mean_file $_ --lambda $lamb --thresh 0.5 --output ${res.name}_${lamb}.csv
    """
}

process DistVal_NEE {

    clusterOptions = "-S /bin/bash"
//    publishDir TENSORBOARDDIST_NEE, mode: "copy", overwrite: true
    maxForks = 2

    input:
    file path from DISTANCE_FOLD6_NEE
    file py from DISTVAL
    val home from params.home
    file _ from MeanFile7_NEE
    file __ from ValRECORDDISTUNET_NEE
    file res from RES_DIST_NEE
    each p from P1
    output:
    file "${res.name}_${p}.csv" into DIST_VAL_NEE

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --path $path  --log ${res} --batch_size 1 --n_features ${res.name.split('_')[0]} --mean_file $_ --thresh 0.9 --output ${res.name}_${p}.csv
    """
}