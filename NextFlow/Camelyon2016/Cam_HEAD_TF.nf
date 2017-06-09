
params.inputfolder = "/share/data40T_v2/CAMELYON16_data/small_temp"
TIFF_TUMOR = file(params.inputfolder + "/Tumor/*.tif")
TIFF_NORMAL  = file(params.inputfolder + "/Normal/*.tif")
ROI_PY = file('ChopPatient.py')
ALL_FOLDER = "/share/data40T_v2/CAMELYON16_precut"


process ChopNormalPatient {

    executor 'sge'
    profile = 'cluster'
    queue = "all.q"
    validExitStatus 0
    clusterOptions = "-S /bin/bash -l mem_free=1G"
    publishDir ALL_FOLDER, overwrite: false
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
    publishDir ALL_FOLDER, overwrite: false
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
    """
}
