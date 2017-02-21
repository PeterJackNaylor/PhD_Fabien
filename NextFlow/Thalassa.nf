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

remote_DATA_FOLDER = WD_REMOTE/DATA_FOLDER

process PriorJob {
    profile = 'cluster'

    input:
    file PYTHONFILE from DISTRIBUTED_VERSION
    file x from TIFF_REMOTE
    val wd_REMOTE from WD_REMOTE

    output:
    file "$wd/PatientFolder/$x/PredOneSlide.sh" into JOB_SUBMIT
    file $x into TIFF_REMOTE

    """
    OUTPUT=$wd/PatientFolder/$x.name/
    METHOD=grid_etienne

 
    python $PYTHONFILE --slide $x --output \$OUTPUT --method \$METHOD --tc 10 --size_tiles 224
    """
}
    
    
