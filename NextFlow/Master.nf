#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/
params.in = "/media/pnaylor/Peter-Work/Projet_FR-TNBC-2015-09-30/newslides/*"
TIFF_LOCAL = file(params.in)

HOST_NAME = "thalassa"

WD_REMOTE = "/share/data40T_v2/Peter"
WD_LOCAL  = "/home/pnaylor/Documents/RemoteComputer"

DATA_FOLDER = "Data"




remote_DATA_FOLDER = WD_REMOTE/DATA_FOLDER
local_DATA_FOLDER = WD_LOCAL/DATA_FOLDER

process SendToCluster {
    input:
    file x from TIFF_LOCAL
    val host from HOST_NAME
    val FILE_PATH_HOST from remote_DATA_FOLDER

    output:
    val "$FILE_PATH_HOST/$x" into TIFF_REMOTE
    """

    FINALDESTINATION=$FILE_PATH_HOST/Biopsy/$x

    if ssh $host stat \$FINALDESTINATION 1> /dev/null 2>&1
            then
                echo "$x has already been copied";
            else
                scp $x $host:\$FINALDESTINATION;

    fi
    """
}


DISTRIBUTED_THAL = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/')


process PriorJob {

    input:
    file PYTHONFILE from DISTRIBUTED_VERSION
    file x from TIFF_REMOTE
    val wd_REMOTE from WD_REMOTE
    
    output:
    file "$wd/PatientFolder/$x/PredOneSlide.sh" into JOB_SUBMIT

    """

    ssh thalassa ""

    """
}
    
    