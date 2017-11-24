
// General parameters
params.image_dir = '/data/users/pnaylor/Bureau'
IMAGE_FOLD = file(params.image_dir + "/ForDataGenTrainTestVal")


/*          0) Resave all the images so that they have 1 for label instead of 255 
TODO: include size_test to tfrecords make values in tfrecords go to 1 and not 255!
In outputs:
newpath name
*/

process ChangeInput {



}


/*          1) We create all the needed records 
In outputs:
a set with the name, the split and the record
*/

TFRECORDS = file('src/TFRecords.py')
UNET_RECORDS = ["UNet", "--UNet", 212]
FCN_RECORDS = ["FCN", "--no-UNet", 224]
DIST_RECORDS = ["DIST", "--no-UNet", 224]
RECORDS_OPTIONS = [UNET_RECORDS, FCN_RECORDS, DIST_RECORDS]
RECORDS_HP = [["train", "16", ""], ["test", "1", 500], ["validation", "1", 1000]]

process CreateRecords {

    file py from TFRECORDS
    val epoch from params.epoch
    file path from IMAGE_FOLD
    set name, unet, size_train from RECORDS_OPTIONS
    each set split, crop, size_test from RECORDS_HP

    output:
    set "$name", "$split", file("${split}_${name}.tfrecords") into NSR0, NSR1, NSR2
    """
    python $py --output ${split}_${name}.tfrecords --split $split --path $path --crop $crop $unet --size_train $size_train --size_test $size_train --seed 42 --epoch $epoch --type JUST_READ 
    """
}
NSR0.filter( it[1] == "train" ).set{TRAIN_REC}
NSR1.filter( it[1] == "test" ).set{TEST_REC}
NSR2.filter( it[1] == "validation" ).set{VAL_REC}

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
}

process TrainingFCN {





}