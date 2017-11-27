
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
NAMES = ["FCN", "UNet"]

process ChangeInput {

    input:
    file path from IMAGE_FOLD
    file changescale from CHANGESCALE
    each name from NAMES
    output:
    set val("$name"), file("ImageFolder") into IMAGE_FOLD2, IMAGE_FOLD3, IMAGE_FOLD4
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
    set val("DIST"), file("ToAnnotateDistance") into DISTANCE_FOLD

    """
    python $py $toannotate
    """
}

/*          1) We create all the needed records 
In outputs:
a set with the name, the split and the record
*/

TFRECORDS = file('src/TFRecords.py')
IMAGE_FOLD2 .concat(DISTANCE_FOLD) .set{FOLDS}
IMAGE_FOLD3 .concat(DISTANCE_FOLD).set{FOLDS2}
UNET_RECORDS = ["UNet", "--UNet", 212]
FCN_RECORDS = ["FCN", "--no-UNet", 224]
DIST_RECORDS = ["DIST", "--UNet", 212]
RECORDS_OPTIONS = Channel.from(UNET_RECORDS, FCN_RECORDS, DIST_RECORDS)
FOLDS.join(RECORDS_OPTIONS) .set{RECORDS_OPTIONS_v2}
RECORDS_HP = [["train", "16", "0"], ["test", "1", 500]]

process CreateRecords {

    input:
    file py from TFRECORDS
    val epoch from params.epoch
    set name, file(path), unet, size_train from RECORDS_OPTIONS_v2
    each op from RECORDS_HP

    output:
    set val("${name}"), val("${op[0]}"), file("${op[0]}_${name}.tfrecords") into NSR0, NSR1, NSR2
    """
    python $py --tf_record ${op[0]}_${name}.tfrecords --split ${op[0]} --path $path --crop ${op[1]} $unet --size_train $size_train --size_test ${op[2]} --seed 42 --epoch $epoch --type JUST_READ 
    """
}
NSR0.filter{ it -> it[1] == "train" }.set{TRAIN_REC}
NSR1.filter{ it -> it[1] == "test" }.set{TEST_REC}

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
a set with the name, the parameters of the model
*/

ITERTEST = 24

ITER8 = 108 // 00


LEARNING_RATE = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
FEATURES = [16, 32, 64]
WEIGHT_DECAY = [0.00005, 0.0005]
BS = 10

Unet_file = file('src/UNet.py')
Fcn_file = file('src/FCN.py')
Dist_file = file('src/Dist.py')

UNET_TRAINING = ["UNet", Unet_file, 212, 0]
FCN_TRAINING  = ["FCN", Fcn_file, 224, ITER8]
DIST_TRAINING = ["DIST", Dist_file, 212, 0]

TRAINING_CHANNEL = Channel.from(UNET_TRAINING, FCN_TRAINING, DIST_TRAINING)
PRETRAINED_8 = file(params.image_dir + "/pretrained/checkpoint16/")

TRAIN_REC.join(TRAINING_CHANNEL).join(FOLDS2) .set {TRAINING_OPTIONS}

process Training {
    maxForks 2

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"
    input:
    set name, split, file(rec), file(py), size, iters, file(path) from TRAINING_OPTIONS
    val home from params.home
    val bs from BS
    each feat from FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY    
    file _ from MeanFile
    file __ from PRETRAINED_8
    val epoch from params.epoch
    output:
    set val("$name"), file("${name}__${feat}_${wd}_${lr}", file("$py"), feat, wd, lr into RESULT_TRAIN

    when:
    "$name" != "FCN" || ("$feat" == "${FEATURES[0]}" && "$wd" == "${WEIGHT_DECAY[0]}")


    script:
    """
    python $py --tf_record $rec --path $path  --log ${name}__${feat}_${wd}_${lr} --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file $_ --n_threads 100 --restore $__ --size_train $size --split $split --iters $iters
    """

} 

/*          3) We test
In inputs: Meanfile, name, split, rec

In outputs:
a set with the name, the split and the record
*/

RESULT_TRAIN .join(TEST_REC) .set {TEST_OPTIONS}

process Testing {
    maxForks 2

    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"
    input:
    set name, file(model), file(py), feat, wd, lr, file(rec), split from TEST_OPTIONS    
    file _ from MeanFile

    output:
    set val("$name"), file("${name}__${feat}_${wd}_${lr}.csv") into RESULT_TRAIN

    script:
    """
    python $py --tf_record $rec --path $path  --log $model --batch_size 1 --n_features $feat --mean_file $_ --n_threads 100 --size_train $size --split $split --iters $iters --split $split
    """  

}

