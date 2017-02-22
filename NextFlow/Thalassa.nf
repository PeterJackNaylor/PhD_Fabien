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





process SendToCluster {
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false

    input:
    file PYTHONFILE from DISTRIBUTED_VERSION
    file x from TIFF_REMOTE
    val wd_REMOTE from WD_REMOTE

    output:
    file "PatientFolder/Job_$x" into JOB_SUBMIT

    """

    FOLDER=`echo $x.name | cut -d '.' -f1`
    OUTPUT=$wd_REMOTE/PatientFolder/$x
    METHOD=grid_etienne

 
    python $PYTHONFILE --slide $x --output PatientFolder/Job_$x --method \$METHOD --tc 10 --size_tiles 224

    """
}