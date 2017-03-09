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

    output:
    file "PatientFolder/Job_${x.getBaseName()}" into JOB_SUBMIT
    file "PatientFolder/Job_${x.getBaseName()}/ParameterDistribution.txt" into PARAM_JOB

    """
    METHOD=grid_etienne
 
    python $PYTHONFILE --slide $x --output PatientFolder/Job_${x.getBaseName()} --method \$METHOD --tc 10 --size_tiles 224

    """
}

process AnalyseEachChop {
    executor 'local'
    profile = 'cluster'

    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false

    input:
    file param_job_txt from PARAM_JOB
    file IMAGE_ANALYSER
    file CBS
    val biopsy_fold from BIOPSY_FOLD
    file nextflow_cfg

    output:
    file "${param_job_txt.getParent()}" into JOBS
    """

    parents=`readlink -f $param_job_txt`
    parents=\${parents%/*}
    ln -s \$parents/PredictionSlide.py PredictionSlide.py
    slide=\${parents##*/}
    slide=\${slide##_*}

    slide=\$(echo \$slide | cut -d'_' -f2)



    parents_biopsy=`readlink -f $biopsy_fold`
    ln -s \$parents_biopsy/\$slide.tiff \$slide


    nextflow $IMAGE_ANALYSER --slideName \$slide --parametertext $param_job_txt --CBS $CBS --py PredictionSlide.py
    """
}
