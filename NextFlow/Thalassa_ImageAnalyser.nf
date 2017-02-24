#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/


params.folder = "/share/data40T_v2/Peter/PatientFolder/*"
PATIENT = file(params.in)

params.text = "/share/data40T_v2/Peter/PatientFolder/*/ParameterDistribution"
TEXT = file(params.in)



HOST_NAME = "thalassa"

WD_REMOTE = "/share/data40T_v2/Peter"

DATA_FOLDER = "Data"

DISTRIBUTED_VERSION = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')




process AllJobs {

	profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false


    input:
    file "param_job" from TEXT.splitText()
    file folder from PATIENT

    """
    FIELD0=`echo $param_job | cut -d' ' -f2`
    FIELD1=`echo $param_job | cut -d' ' -f3`
    FIELD2=`echo $param_job | cut -d' ' -f4`
    FIELD3=`echo $param_job | cut -d' ' -f5`
    FIELD4=`echo $param_job | cut -d' ' -f6`

    OUTPUT=$folder
    SIZE=224
    PYTHONFILE=$folder/PredictionSlide.py

    nextflow CheckingBeforeSubmit.nf --py \$PYTHON_FILE --x \$FIELD0 --y \$FIELD1 --size_x \$FIELD2 --size_y \$FIELD3 --ref_level \$FIELD4 --output \$OUTPUT --slide \$SLIDE --size \$SIZE -profiles cluster -resume

    """


}

