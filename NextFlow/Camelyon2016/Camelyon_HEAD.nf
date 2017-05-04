#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/
params.inputfolder = "/share/data40T_v2/CAMELYON16_data"
TIFF_TUMOR = file(params.inputfolder + "/Tumor/*.tif")
TIFF_NORMAL  = file(params.inputfolder + "/Normal/*.tif")

HOST_NAME = "thalassa"

WD_REMOTE = file("./OUTPUT")
WD_REMOTE_VAL = "/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/NextFlow/Camelyon2016/OUTPUT"
WD_DG = file('./OUTPUT_DG')
TXT_FILES_OUTPUT_1PROCESS = file('./OUTPUT/*.txt')
WD_LOCAL  = "/home/pnaylor/Documents/RemoteComputer"

DATA_FOLDER = "Data"


ROI_PY = file('ChopPatient.py')
MAKEPROTOTXT = file("MakePrototxt.py")
MAKESOLVER = file("MakeSolver.py")

TRAINTESTPY = file("TrainTestSet.py")
MAKEDATAGEN = file("MakeDataGen.py")
TRAIN_PY = file("TrainingWithSaves.py")
LEVELDB = file("WriteLevelDB.py")

LR = 0.001
MOMENTUM = 0.99
WEIGHT_DECAY = 0.00005
GAMMA = 0.1
STEPSIZE = 100000

SPLIT_RATIO = 0.5
NBER_PATIENTS = 30
TICKET_VAL = 60
BS = 4
N_ITER = 1000
DISP_INTERVAL = 100
NUMBER_OF_TEST = 1


process ChopNormalPatient {

    executor 'sge'
    profile = 'cluster'
    queue = "all.q"
    validExitStatus 0
    clusterOptions = "-S /bin/bash -l mem_free=1G"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 100
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file normal from TIFF_NORMAL
    file pythonfile from ROI_PY
    val ticket_val from TICKET_VAL

    output:
    file "${normal.getBaseName()}.txt" into DB_N
    file "*.png" into DB_N_IMG
    
    """
    python $ROI_PY --output . --type 'Normal' --file $normal --ticket_val $ticket_val
    """
}

process ChopTumorPatient {

    executor 'sge'
    profile = 'cluster'
    queue = "all.q"
    validExitStatus 0
    clusterOptions = "-S /bin/bash -l mem_free=1G"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 100
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file tumor from TIFF_TUMOR
    file pythonfile from ROI_PY

    output:
    file "${tumor.getBaseName()}.txt" into DB_T
    file "*.png" into DB_T_IMG

    """
    python $ROI_PY --output . --type 'Tumor' --file $tumor
    """
}



process CreateTrainTestSet {
    executor 'local'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash -l h_vmem=1G"
    publishDir WD_DG, mode: "copy", overwrite: false
    maxForks = 1

    input:
    val nber_patient from NBER_PATIENTS
    val split_value from SPLIT_RATIO
    val txt_fold from WD_REMOTE_VAL
    file traintestset from TRAINTESTPY
    file makedatagen from MAKEDATAGEN
    file wait from DB_N .mix(DB_T) .toList()
    file pathfolder from WD_DG

    output:
    file "train.txt" into DG_TRAIN, DG_TRAIN2
    file "test.txt" into DG_TEST, DG_TEST2
    file "train.pkl" into DG_TRAINING, DG_TRAINING2
    file "test.pkl" into DG_TESTING, DG_TESTING2
    file "my_little_helper_train.txt" into DG_HELPER_TRAIN
    file "my_little_helper_test.txt" into DG_HELPER_TEST


    script:
    """
    python $traintestset --input $txt_fold --output . --nber_patient $nber_patient --split_value $split_value
    python $makedatagen --input train.txt --output train.pkl --split train --pathfolder $txt_fold
    python $makedatagen --input test.txt  --output test.pkl  --split test --pathfolder $txt_fold
    """
}

/* check if data gen still works as plan */ 
process CreateTrainPrototxt {
    executor 'local'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_DG, overwrite: false
    maxForks = 2

    input:
    file makeprototxt from MAKEPROTOTXT
    val bs from BS
    file train_pkl from DG_TRAINING

    output:
    file "train.prototxt" into TRAIN_PROTOTXT, TRAIN_PROTOTXT2

    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    script:
    """ 
    python $makeprototxt --dg $train_pkl --split train --cn trainCAM16 --loss softmax --batch_size $bs --num_output 2 -o .
    """
}

process CreateTestPrototxt {

    executor 'local'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_DG, overwrite: false
    maxForks = 2

    input:
    file makeprototxt from MAKEPROTOTXT
    file test_pkl from DG_TESTING

    output:
    file "test.prototxt" into TEST_PROTOTXT, TEST_PROTOTXT2
    
    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    script:
    """ 
    python $makeprototxt --dg $test_pkl --split test  --cn testCAM16  --loss softmax --batch_size 1   --num_output 2 -o .
    """
}

process CreateSolver {

    executor 'local'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_DG, overwrite: false
    maxForks = 1

    input:
    file makesolver from MAKESOLVER
    file train_prototxt from TRAIN_PROTOTXT
    file test_prototxt from TEST_PROTOTXT
    val lr from LR
    val momentum from MOMENTUM
    val weight_decay from WEIGHT_DECAY
    val gamma from GAMMA 
    val stepsize from STEPSIZE

    output:
    file "solver.prototxt" into SOLVER_PROTOTXT
    file "snapshot"
    file train_prototxt into TRAIN_PROT
    file test_prototxt into TEST_PROT

    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    script:
    """ 
    python $makesolver --trainnet $train_prototxt --testnet $test_prototxt --lr $lr --momentum $momentum --weight_decay $weight_decay --gamma $gamma --stepsize $stepsize
    """
}

 

process CreateLMDB_test {

    executor 'local'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_DG, overwrite: false
    maxForks = 1

    input:
    file writeleveldb from LEVELDB
    file test_pkl from DG_TESTING2
    file DG_HELPER_TEST
    file DG_TEST2

    output:
    file "test.lmdb" into TEST_LMDB, TEST_LMDB2
    
    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    script:
    """ 

    python $writeleveldb --lmdb_file test.lmdb --bs 1000 --datagen $test_pkl 
    """
}

process CreateLMDB_train {

    executor 'local'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_DG, overwrite: false
    maxForks = 1

    input:
    file TEST_LMDB2
    file writeleveldb from LEVELDB
    file train_pkl from DG_TRAINING2
    file DG_HELPER_TRAIN
    file DG_TRAIN2

    output:
    file "train.lmdb" into TRAIN_LMDB
    
    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    script:
    """ 
    python $writeleveldb --lmdb_file train.lmdb --bs 1000 --datagen $train_pkl 
    """
}


process Training {

    executor 'sge'
    queue = "cuda.q"
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_DG, overwrite: false
    maxForks = 2

    input:
    file train from TRAIN_PY
    file solver from SOLVER_PROTOTXT
    val n_iter from N_ITER
    val disp_interval from DISP_INTERVAL
    val number_of_test from NUMBER_OF_TEST
    file train_prototxt from TRAIN_PROTOTXT2
    file test_prototxt from TEST_PROTOTXT2
    file train_lmdb from TRAIN_LMDB
    file test_lmdb from TEST_LMDB
//    file test_pkl from DG_TESTING2
//    file DG_HELPER_TEST
//    file DG_TEST2
//    file train_pkl from DG_TRAINING2
//    file DG_HELPER_TRAIN
//    file DG_TRAIN2

    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter_crf_cbio/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    """

    python $train --solver $solver --wd . --cn testCAM16 --n_iter $n_iter --disp_interval $disp_interval --number_of_test $number_of_test --num testCAM16
    """
}