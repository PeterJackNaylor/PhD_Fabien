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

WD_REMOTE = "./OUTPUT"
WD_LOCAL  = "/home/pnaylor/Documents/RemoteComputer"

DATA_FOLDER = "Data"





ROI_PY = file('ChopPatient.py')


process ChopNormalPatient {

	executor 'sge'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, mode: "move", overwrite: false
    maxForks = 5


    input:
    file normal from TIFF_NORMAL
    file pythonfile from ROI_PY


    output:
    file "${normal.getBaseName()}.txt" into DB_N


    """

    python $ROI_PY --output . --type 'Normal' --file $normal

    """


}


process ChopTumorPatient {

    executor 'sge'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, mode: "move", overwrite: false
    maxForks = 5


    input:
    file tumor from TIFF_TUMOR
    file pythonfile from ROI_PY


    output:
    file "${tumor.getBaseName()}.txt" into DB_T


    """

    python $ROI_PY --output . --type 'Tumor' --file $tumor

    """


}

SPLIT_RATIO = 0.5
NBER_PATIENTS = 30
TEXTOUTPUT = "/share/data40T_v2/Peter/PythonScript/PhD_Fabien/NextFlow/Camelyon2016/"
TRAINTESTPY = "TrainTestSet.py"


process CreateTrainTestSet {
    executor 'local'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, mode: "move", overwrite: false
    maxForks = 5


    input:
    val nber_patient from NBER_PATIENTS
    val split_value from SPLIT_VALUE
    file txt_fold from TEXTOUTPUT
    file traintestset from TRAINTESTPY
    output:
    file "Train.txt" into DG_TRAIN
    file "Test.txt" into DG_TEST


    when:
    TEXTOUTPUT.size() == (TIFF_NORMAL.size() + TIFF_TUMOR.size())


    """

    python $TrainTestSet --input $txt_fold --output . --nber_patient $nber_patient --split_value $split_value
    
    """
}


process CreateDataGenerator {
    executor 'local'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, mode: "move", overwrite: false
    maxForks = 5


    input:
    file train_txt from DG_TRAIN
    file test_txt from DG_TEST
    output:
    file "train.pkl" into DG_TRAINING
    file "test.pkl" into DG_TESTING

    """

    python $TrainTestSet --input $txt_fold --output . --nber_patient $nber_patient --split_value $split_value
    
    """
}

