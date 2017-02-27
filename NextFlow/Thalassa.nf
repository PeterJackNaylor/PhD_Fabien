#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/

params.in = "/share/data40T_v2/Peter/Data/Biopsy/*"
TIFF_REMOTE = file(params.in)

HOST_NAME = "thalassa"

WD_REMOTE = "/share/data40T_v2/Peter"

DATA_FOLDER = "Data"

DISTRIBUTED_VERSION = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')

IMAGE_ANALYSER = file("Thalassa_ImageAnalyser.nf")
CBS = file("CheckingBeforeSubmit.nf")

process ChopPatient {
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
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false

    input:
    file param_job_txt from PARAM_JOB
    file IMAGE_ANALYSER
    file CBS
    """

    PatientPath=TempOutput
    parents=`readlink -f $param_job_txt`
    parents=\${parents%/*}
    slide=\${parents##*/}
    slide=\${slide##_*}
    echo \$slide
    nextflow $IMAGE_ANALYSER --folder \$PatientPath --slideName \$slide --text $param_job_txt --CBS $CBS -profile cluster -resume
    """
}