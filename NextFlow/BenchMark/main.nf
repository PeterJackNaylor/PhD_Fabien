
// General parameters
params.image_dir = '/data/users/pnaylor/Bureau'
params.epoch = 1
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"
IMAGE_FOLD = file(params.image_dir + "/ForDataGenTrainTestVal")

/*          0) a) Resave all the images so that they have 1 for label instead of 255 
            0) b) Resave all the images so that they are distance map
            0) c) Resave all the images so that they are normalized
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
    set val("$name"), file("ImageFolder") into IMAGE_FOLD2, IMAGE_FOLD3 
    """
    python $changescale --path $path

    """
}

BinToDistanceFile = file('src/BinToDistance.py')

process BinToDistance {
    input:
    file py from BinToDistanceFile
    file toannotate from IMAGE_FOLD
    output:
    set val("DIST"), file("ToAnnotateDistance") into DISTANCE_FOLD, DISTANCE_FOLD2

    """
    python $py $toannotate
    """
}

REFERENCE = file(IMAGE_FOLD + '/Slide_test/test_{1,2}.png')
MAT_NORMALIZE = file('src/Normalize.m')
TYPE = ["Macenko", "RGBHist"]
TOOL_BOX = '/Data/users/pnaylor/Documents/MATLAB/stain_normalisation_toolbox/'

process Normalise {
    publishDir "${type}_images"
    input:
    set name, file(path) from DISTANCE_FOLD
    file matlab_n from MAT_NORMALIZE
    file reference from REFERENCE
    each type from TYPE
    val tool_box from TOOL_BOX
    output:
    set val("DIST_${type}_${reference.baseName.split('_')[1]}"), file("ImageFolder") into NORM_FOLD
    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_n.baseName} $path $type $reference $tool_box;exit;'
    """
}

/*          1) We create all the needed records 
In outputs:
a set with the name, the split and the record
*/

TFRECORDS = file('src/TFRecords.py')
IMAGE_FOLD2 .concat(DISTANCE_FOLD) .concat(NORM_FOLD) .into{FOLDS;FOLDS2}
UNET_REC = ["UNet", "--UNet", 212]
FCN_REC = ["FCN", "--no-UNet", 224]
DIST_REC = ["DIST", "--UNet", 212]
MACENKO_1 = ["DIST_Macenko_1", "--UNet", 212]
MACENKO_2 = ["DIST_Macenko_2", "--UNet", 212]
RGBHIST_1 = ["DIST_RGBHist_1", "--UNet", 212]
RGBHIST_2 = ["DIST_RGBHist_2", "--UNet", 212]

RECORDS_OPTIONS = Channel.from(UNET_REC, FCN_REC, DIST_REC, MACENKO_1, MACENKO_2, RGBHIST_1, RGBHIST_2)
FOLDS.join(RECORDS_OPTIONS) .set{RECORDS_OPTIONS_v2}
RECORDS_HP = [["train", "16", "0"], ["test", "1", 500], ["validation", "1", 996]]

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
NSR2.filter{ it -> it[1] == "validation" }.set{VAL_REC}

/*          2) We create the mean
In outputs:
a set with the name, the split and the record
*/

MEANPY = file('src/MeanCalculation.py')

process Mean {
    input:
    file py from MEANPY
    set val(name), file(toannotate) from FOLDS2
    output:
    set val("$name"), file("mean_file.npy"), file("$toannotate") into MeanFile, Meanfile2, Meanfile2VAL, Meanfile3, Meanfile3VAL
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
ITER8 = 10800
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

Channel.from(UNET_TRAINING, FCN_TRAINING, DIST_TRAINING) .into{ TRAINING_CHANNEL; TRAINING_CHANNEL2; TRAINING_CHANNELVAL2}
PRETRAINED_8 = file(params.image_dir + "/pretrained/checkpoint16/")
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
    file __ from PRETRAINED_8
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
            4) b) We choose the best hyperparamter with respect to the validation data set

In inputs: Meanfile, image_path resp., split, rec, model, python, feat
In outputs: a set with the name and model or csv
*/
// a)
P1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
P2 = [0.5, 1.0, 1.5, 2.0]
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
    python $py --tf_record $rec --path $path  --log $model --batch_size 1 --n_features $feat --mean_file ${mean} --n_threads 100 --split $split --size_test 500 --p1 ${p1} --p2 ${p2} --restore $model --iters $iters --output ${name}__${feat}_${wd}_${lr}_${p1}_${p2}.csv
    """  

}

// b)
ITERVAL = 14
VAL_REC.cross(RESULT_TRAIN_VAL).map{ first, second -> [first, second.drop(1)].flatten() } .set{ VAL_OPTIONS_pre }
Meanfile2VAL.cross(VAL_OPTIONS_pre).map { first, second -> [first, second.drop(1)].flatten() } .into{VAL_OPTIONS;VAL_OPTIONS2}

process Testing_Val {
    maxForks 2
    beforeScript "source $home/CUDA_LOCK/.whichNODE"
    afterScript "source $home/CUDA_LOCK/.freeNODE"
    input:
    set name, file(mean), file(path), split, file(rec), file(model), file(py), feat, wd, lr from VAL_OPTIONS    
    each p1 from P1
    each p2 from P2
    val iters from ITERVAL
    val home from params.home
    output:
    set val("$name"), file("${name}__${feat}_${wd}_${lr}_${p1}_${p2}.csv") into RESULT_VAL
    set val("$name"), file("$model") into MODEL_TEST
    when:
    ("$name" =~ "DIST" && p1 < 6) || ( !("$name" =~ "DIST") && p2 == P2[0] && p1 > 5)
    script:
    """
    python $py --tf_record $rec --path $path  --log $model --batch_size 1 --n_features $feat --mean_file ${mean} --n_threads 100 --split $split --size_test 996 --p1 ${p1} --p2 ${p2} --restore $model --iters $iters --output ${name}__${feat}_${wd}_${lr}_${p1}_${p2}.csv
    """  

}


/*          5) We regroup a) the test on dataset 1
                          b) the test on dataset 2 (same as for val)
In inputs: name, all result_test.csv per key
In outputs: name, best_model, p1, p2
*/
// a)
REGROUP = file('src/regroup.py')
RESULT_TEST  .groupTuple() 
             .set { KEY_CSV }
RESULT_TRAIN2.map{name, model, py, feat, wd, lr -> [name, model]} .groupTuple() . set {ALL_MODELS}
KEY_CSV .join(ALL_MODELS) .set {KEY_CSV_MODEL}

process GetBestPerKey {
    publishDir "./Test_tables/" , pattern: "*.csv"
    input:
    file py from REGROUP
    set name, file(csv), file(model) from KEY_CSV_MODEL

    output:
    set val("$name"), file("best_model") into BEST_MODEL_TEST
    file 'feat_val' into N_FEATS
    file 'p1_val' into P1_VAL
    file 'p2_val' into P2_VAL
    file "${name}_test.csv"
    """
    python $py --store_best best_model --output ${name}_test.csv
    """
}

// b)
RESULT_VAL  .groupTuple() 
             .set { KEY_CSV_VAL }
RESULT_TRAIN_VAL2.map{name, model, py, feat, wd, lr -> [name, model]} .groupTuple() . set {ALL_MODELS_VAL}
KEY_CSV_VAL .join(ALL_MODELS_VAL) .set {KEY_CSV_MODEL_VAL}

process GetBestPerKey_Val {
    publishDir "./Val_tables/" , pattern: "*.csv"
    input:
    file py from REGROUP
    set name, file(csv), file(model) from KEY_CSV_MODEL_VAL

    output:
    set val("$name"), file("best_model") into BEST_MODEL_TEST_VAL
    file 'feat_val' into N_FEATS_VAL
    file 'p1_val' into P1_VAL_VAL
    file 'p2_val' into P2_VAL_VAL
    file "${name}_val.csv"
    """
    python $py --store_best best_model --output ${name}_val.csv
    """
}

/*
Compute validation score on validation set
a) Validation with hyper parameter choosen on different dataset
b) Validation with hyper parameter choosen on the same dataset

*/
// a)
BEST_MODEL_TEST.join(TRAINING_CHANNEL2).join(Meanfile3) .set{ VALIDATION_OPTIONS}
N_FEATS .map{ it.text } .set {FEATS_}
P1_VAL  .map{ it.text } .set {P1_}
P2_VAL  .map{ it.text } .set {P2_}

process Validation {
    publishDir "./Validation/"
    input:
    set name, file(best_model), file(py), _, __, file(mean), file(path) from VALIDATION_OPTIONS
    val feat from FEATS_ 
    val p1 from P1_
    val p2 from P2_
    output:
    file "./$name"
    file "${name}.csv" into CSV_VAL
    """
    python $py --mean_file $mean --path $path --log $best_model --restore $best_model --batch_size 1 --n_features ${feat} --n_threads 100 --split validation --size_test 500 --p1 ${p1} --p2 ${p2} --output ${name}.csv --save_path $name
    """
}

// b)
BEST_MODEL_TEST_VAL.join(TRAINING_CHANNELVAL2).join(Meanfile3VAL) .set{ VALIDATION_OPTIONS_VAL}
N_FEATS_VAL .map{ it.text } .set {FEATS_VAL}
P1_VAL_VAL  .map{ it.text } .set {P1_VAL}
P2_VAL_VAL  .map{ it.text } .set {P2_VAL}

process Validation_VAL {
    publishDir "./ValidationWithoutNested/"
    input:
    set name, file(best_model), file(py), _, __, file(mean), file(path) from VALIDATION_OPTIONS_VAL
    val feat from FEATS_VAL
    val p1 from P1_VAL
    val p2 from P2_VAL
    output:
    file "./$name"
    file "${name}.csv" into CSV_VAL_VAL
    """
    python $py --mean_file $mean --path $path --log $best_model --restore $best_model --batch_size 1 --n_features ${feat} --n_threads 100 --split validation --size_test 500 --p1 ${p1} --p2 ${p2} --output ${name}.csv --save_path $name
    """
}

PLOT = file('src/plot.py')

process Plot {
    publishDir "./Validation/"
    input:
    file _ from CSV_VAL .collect()
    file py from PLOT
    output:
    file "BarResult_train_test_val.png"
    """
    python $py --output BarResult_train_test_val.png --output_csv Result_train_test_val.csv
    """
}