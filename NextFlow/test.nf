#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/


params.test = "Eureka"

VALUE = params.test

WD_REMOTE = "/share/data40T_v2/Peter"

params.CBS = "TestInside.nf"
EURA = file("eureka.txt")
CBS = file(params.CBS)
nextflow_cfg = file("nextflow.config")


process FirstNF {
    executor 'local'
	profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, mode: "copy", overwrite: false


    input:
    file CBS
    file nextflow_cfg
    val file from VALUE
    file EURA
    output:
    file "test.txt" into TEST_OUT

    """
    nextflow run $CBS --text eureka.txt --value $file  -profile cluster -resume
    """


}

