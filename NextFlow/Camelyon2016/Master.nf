#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/
params.inputfolder = "/share/data40T_v2/CAMELYON16_data"
TIFF_NORMAL = file(params.inputfolder + "/Tumor/*.tiff")
TIFF_TUMOR  = file(params.inputfolder + "/Normal/*.tiff")

HOST_NAME = "thalassa"

WD_REMOTE = "/share/data40T_v2/Peter"
WD_LOCAL  = "/home/pnaylor/Documents/RemoteComputer"

DATA_FOLDER = "Data"





ROI_PY = file('ChopPatient.py')


process ChopNormalPatient {

	executor 'sge'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 5


    input:
    file normal from TIFF_NORMAL
    file pythonfile from ROI_PY


    output:
    file ${normal.getBaseName()}.txt

    """

    python ChopPatient --type --name --

    """




}