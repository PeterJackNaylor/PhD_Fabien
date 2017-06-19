#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/

params.in = "/share/data40T_v2/Peter/Data/Biopsy/"
TIFF_REMOTE = file(params.in + "*")
BIOPSY_FOLD = params.in
HOST_NAME = "thalassa"

MARGE = 100
WD_REMOTE = "/share/data40T_v2/Peter"

DATA_FOLDER = "Data"

DISTRIBUTED_VERSION = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')

IMAGE_ANALYSER = file("Thalassa_ImageAnalyser.nf")

CBS = file("CheckingBeforeSubmit.nf")

nextflow_cfg = file("nextflow.config")

process ChopPatient {
    executor 'sge'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false

    input:
    file PYTHONFILE from DISTRIBUTED_VERSION
    file x from TIFF_REMOTE
    val wd_REMOTE from WD_REMOTE
    val marge from MARGE
    output:
    file "Job_${x.getBaseName()}" into JOB_SUBMIT
    file "Job_${x.getBaseName()}/ParameterDistribution.txt" into PARAM_JOB
    val "$marge" into MARGE2
    """
    METHOD=grid_etienne
 
    python $PYTHONFILE --slide $x --output Job_${x.getBaseName()} --method \$METHOD --tc 10 --size_tiles 224 --marge $marge

    """
}

ALL_CONFIG = Channel.fromPath('/share/data40T_v2/Peter/PatientFolder/Job_*/ParameterDistribution.txt')
                    .splitText()


process subImage {
    executor 'sge'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false


    input:
    val param_job_txt from ALL_CONFIG
    file param from PARAM_JOB.first()
    val inputt from params.in
    val marge from MARGE2.first()

    output:
    file "Job_*/tiled/*.tiff" into IMAGE_PROCESSED
    file "Job_*/table/*.npy" into TABLE_PROCESSED
    """

    FIELD0=`echo '$param_job_txt' |cut -d' ' -f2`
    FIELD1=`echo '$param_job_txt' |cut -d' ' -f3`
    FIELD2=`echo '$param_job_txt' |cut -d' ' -f4`
    FIELD3=`echo '$param_job_txt' |cut -d' ' -f5`
    FIELD4=`echo '$param_job_txt' |cut -d' ' -f6`
    FIELD5=`echo '$param_job_txt' |cut -d' ' -f7`

    ln -s /share/data40T_v2/Peter/PatientFolder/Job_\$FIELD5/PredictionSlide.py PredictionSlide.py
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_\$FIELD5 Job_\$FIELD5

    python PredictionSlide.py -x \$FIELD0 -y \$FIELD1 --size_x \$FIELD2 --size_y \$FIELD3 --ref_level \$FIELD4 --output PatientFolder/Job_\$FIELD5/ --slide $inputt\$FIELD5.tiff --size 224 --marge $marge

    """


}

process GetMax {

    
}