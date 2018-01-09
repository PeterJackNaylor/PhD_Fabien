
// General parameters
params.image_dir = '/data/users/pnaylor/Bureau'
params.epoch = 1
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
IMAGE_FOLD = file(params.image_dir + "/ForBestSegmentationModel")

/*          0) a) Resave all the images so that they have 1 for label instead of 255 
            0) b) Resave all the images so that they are distance map
            0) c) Resave all the images so that they are normalized
In outputs:
newpath name
*/

NAMES = ["FCN", "UNet"]

BinToDistanceFile = file('src/BinToDistance.py')

process BinToDistance {
    input:
    file py from BinToDistanceFile
    file toannotate from IMAGE_FOLD
    output:
    set val("DIST"), file("ToAnnotateDistance") into DISTANCE_FOLDS, DISTANCE_FOLDS2

    """
    python $py $toannotate
    """
}

/*          1) We create all the needed records 
In outputs:
a set with the name, the split and the record
*/

TFRECORDS = file('src/TFRecords.py')
DIST_REC = ["DIST", "--UNet", 212]

RECORDS_OPTIONS = Channel.from(DIST_REC)
DISTANCE_FOLDS.join(RECORDS_OPTIONS) .set{RECORDS_OPTIONS_v2}
RECORDS_HP = [["train", "4", "0"], ["test", "1", 500]]

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
    set val(name), file(toannotate) from DISTANCE_FOLDS2
    output:
    set val("$name"), file("mean_file.npy"), file("$toannotate") into MeanFile, Meanfile2, Meanfile3
    """
    python $py --path $toannotate --output .
    """
}


/*          3) We train
In inputs: Meanfile, name, split, rec
In outputs:
a set with the name, the parameters of the model
*/

ITERTEST = 50
LEARNING_RATE = [0.001]
FEATURES = [16]
WEIGHT_DECAY = [0.00005]
BS = 10

Dist_file = file('src/Dist.py')

DIST_TRAINING = ["DIST", Dist_file, 212, 0]

Channel.from(DIST_TRAINING) .into{ TRAINING_CHANNEL; TRAINING_CHANNEL2; TRAINING_CHANNELVAL2}
TRAIN_REC.join(TRAINING_CHANNEL).join(MeanFile) .set {TRAINING_OPTIONS}

process Training {
    maxForks 2
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"
    input:
    set name, split, file(rec), file(py), size, iters, file(mean), file(path) from TRAINING_OPTIONS
    val home from params.home
    val bs from BS
    each feat from FEATURES
    each lr from LEARNING_RATE
    each wd from WEIGHT_DECAY   
    val epoch from params.epoch
    output:
    set val("$name"), file("${name}__${feat}_${wd}_${lr}"), file("$py"), feat, wd, lr into RESULT_TRAIN, RESULT_TRAIN2, RESULT_TRAIN_VAL, RESULT_TRAIN_VAL2
    when:
    "$name" != "FCN" || ("$feat" == "${FEATURES[0]}" && "$wd" == "${WEIGHT_DECAY[0]}")
    script:
    """
    python $py --tf_record $rec --path $path  --log ${name}__${feat}_${wd}_${lr} --learning_rate $lr --batch_size $bs --epoch $epoch --n_features $feat --weight_decay $wd --mean_file ${mean} --n_threads 100 --restore $__ --size_train $size --split $split --iters $iters
    """
} 

/*          4) a) We choose the best hyperparamter with respect to the test data set

In inputs: Meanfile, image_path resp., split, rec, model, python, feat
In outputs: a set with the name and model or csv
*/
// a)
P1 = [0, 1]
P2 = [0.5, 1.0]
TEST_REC.cross(RESULT_TRAIN).map{ first, second -> [first, second.drop(1)].flatten() } .set{ TEST_OPTIONS_pre }
Meanfile2.cross(TEST_OPTIONS_pre).map { first, second -> [first, second.drop(1)].flatten() } .into{TEST_OPTIONS;TEST_OPTIONS2}

process Testing {
    maxForks 2
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"
    input:
    set name, file(mean), file(path), split, file(rec), file(model), file(py), feat, wd, lr from TEST_OPTIONS    
    each p1 from P1
    each p2 from P2
    val iters from ITERTEST
    val home from params.home
    output:
    set val("$name"), file("${name}__${feat}_${wd}_${lr}_${p1}_${p2}.csv") into RESULT_TEST
    set val("$name"), file("$model") into MODEL_TEST
    when:
    ("$name" =~ "DIST" && p1 < 6) || ( !("$name" =~ "DIST") && p2 == P2[0] && p1 > 5)
    script:
    """
    python $py --tf_record $rec --path $path  --log $model --batch_size 1 --n_features $feat --mean_file ${mean} --n_threads 100 --split $split --size_test 500 --p1 ${p1} --p2 ${p2} --restore $model --iters $iters --output ${name}__${feat}_${wd}_${lr}_${p1}_${p2}.csv --save_path ${name}__${feat}_${wd}_${lr}_${p1}_${p2}
    """  

}

