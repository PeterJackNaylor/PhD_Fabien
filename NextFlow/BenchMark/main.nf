
// General parameters
params.image_dir = '/data/users/pnaylor/Bureau'
params.epoch = 1
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
IMAGE_FOLD = file(params.image_dir + "/ForDataGenTrainTestVal")

/*          0) Resave all the images so that they have 1 for label instead of 255 
TODO: include size_test to tfrecords make values in tfrecords go to 1 and not 255!
In outputs:
newpath name
*/

CHANGESCALE = file('src/changescale.py')

process ChangeInput {

    input:
    file path from IMAGE_FOLD
    file changescale from CHANGESCALE
    output:
    file "ImageFolder" into IMAGE_FOLD2, IMAGE_FOLD3, IMAGE_FOLD4
    """
    python $changescale --path $path

    """
}

BinToDistanceFile = file('src/BinToDistance.py')

process BinToDistance {
    queue = "all.q"
    clusterOptions = "-S /bin/bash"
    input:
    file py from BinToDistanceFile
    file toannotate from IMAGE_FOLD
    output:
    file "ToAnnotateDistance" into DISTANCE_FOLD

    """
    python $py $toannotate
    """
}

/*          1) We create all the needed records 
In outputs:
a set with the name, the split and the record
*/
//IMAGE_FOLD4 .subscribe{println it}
println( IMAGE_FOLD4 .collect() )
TFRECORDS = file('src/TFRecords.py')
UNET_RECORDS = ["UNet", "--UNet", 212, IMAGE_FOLD2.collect().first()]
FCN_RECORDS = ["FCN", "--no-UNet", 224, IMAGE_FOLD3.collect().first()]
DIST_RECORDS = ["DIST", "--UNet", 212, DISTANCE_FOLD]
RECORDS_OPTIONS = Channel.from([UNET_RECORDS, FCN_RECORDS, DIST_RECORDS])
RECORDS_HP = [["train", "16", "0"], ["test", "1", 500], ["validation", "1", 1000]]

process CreateRecords {

    input:
    file py from TFRECORDS
    val epoch from params.epoch
    set name, unet, size_train, path from RECORDS_OPTIONS
    each op from RECORDS_HP

    output:
    set "${name}", "${op[0]}", file("${op[0]}_${name}.tfrecords") into NSR0, NSR1, NSR2
    """
    python $py --tf_record ${op[0]}_${name}.tfrecords --split ${op[0]} --path $path --crop ${op[1]} $unet --size_train $size_train --size_test ${op[2]} --seed 42 --epoch $epoch --type JUST_READ 
    """
}
NSR0.filter{ it -> it[1] == "train" }.set{TRAIN_REC}
NSR1.filter{ it -> it[1] == "test" }.set{TEST_REC}
NSR2.filter{ it -> it[1] == "validation" }.set{VAL_REC}

/*          2) We create the mean

In outputs:
a set with the name, the split and the record
*/

MEANPY = file('src/MeanCalculation.py')

process Mean {

    input:
    file py from MEANPY
    file toannotate from IMAGE_FOLD
    output:
    file "mean_file.npy" into MeanFile, MeanFile2, MeanFile3, MeanFile4, MeanFile5, MeanFile6, MeanFile7, MeanFile4_NEE, MeanFile7_NEE

    """
    python $py --path $toannotate --output .
    """
}

/*          3) We train
In inputs: Meanfile, name, split, rec

In outputs:
a set with the name, the split and the record
*/

ITERTEST = 24

ITER32 = 10800
ITER16 = 10800
ITER8 = 10800


LEARNING_RATE = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
FEATURES = [16, 32, 64]
WEIGHT_DECAY = [0.00005, 0.0005]
BS = 10

Unet_file = file('src/UNet.py')
Fcn_file = file('src/FCN.py')
Dist_file = file('src/Dist.py')

UNET_TRAINING = ["UNet", Unet_file, 212]
FCN_TRAINING  = ["FCN", Fcn_file, 224]
DIST_TRAINING = ["DIST", Dist_file, 212]

TRAINING_CHANNEL = Channel.from([UNET_TRAINING, FCN_TRAINING, DIST_TRAINING])
PRETRAINED_8 = file(params.image_dir + "/pretrained/checkpoint16/")
/*
TRAIN_REC.join(TRAINING_CHANNEL) .set {TRAINING_OPTIONS}

process Training {

    maxForks = 2

    input:
    set name, file(rec), file(py), size from TRAINING_OPTIONS
    file path from IMAGE_FOLD
    val home from params.home
    val bs from BS
    each feat from FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile
    file __ from PRETRAINED_8
    val epoch from EPOCHUNET
    output:
    set "$name", file("${name}__${feat}_${wd}_${lr}") into RESULT_TRAIN

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"

    script:
    """
    python $py --tf_record $rec --path $path  --log ${name}__${feat}_${wd}_${lr} --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100 --restore $__ 
    """
} */