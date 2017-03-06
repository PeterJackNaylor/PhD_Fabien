#!/usr/bin/env nextflow

/*
Everything is written from the perspective of cookies and crisp
LOCAL is cookies and crisp 
REMOTE is thalassa
*/


params.folder = "/share/data40T_v2/Peter/PatientFolder/*"
PATIENT = file(params.folder)

params.parametertext = "/share/data40T_v2/Peter/PatientFolder/*/ParameterDistribution.txt"
// PARAMETERTEXT = file(params.parametertext)

PARAMETERTEXT = Channel.fromPath(params.parametertext).splitText(by:1)
params.slideName = "/share/data40T_v2/Peter/PatientFolder/555777"
SLIDENAME = file(params.slideName)

params.py = "/share/data40T_v2/Peter/PatientFolder/*/PredictionSlide.py"
PY = file(params.py)

params.CBS = "/share/data40T_v2/Peter/PythonScript/PhD_Fabien/Nextflow/CheckingBeforeSubmit.nf"
CBS = file(params.CBS)
nextflow_cfg = file("nextflow.config")

HOST_NAME = "thalassa"


DATA_FOLDER = "Data"

DISTRIBUTED_VERSION = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')



process getTheFileAndSplitLines {
    input:
    val line from PARAMETERTEXT.splitText()

    output:
    val line into LINES_LIST
    """

    """

}


process CutLine {
    executor 'local'
	profile = 'cluster'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"
    maxForks 10

    input:
    file slide from SLIDENAME
    file cbs from CBS
    file py from PY
    file param_job from LINES_LIST
    file nextflow_cfg


    """
    FIELD0=`cut -d' ' -f2 $param_job`
    FIELD1=`cut -d' ' -f3 $param_job`
    FIELD2=`cut -d' ' -f4 $param_job`
    FIELD3=`cut -d' ' -f5 $param_job`
    FIELD4=`cut -d' ' -f6 $param_job`
    SIZE=224
    PYTHONFILE=PredictionSlide.py
    SLIDE=${slide.getBaseName()}
    
    OUTPUT_FILE=/share/data40T_v2/Peter/PatientFolder/Job_\$SLIDE/tiled/\$FIELD0_\$FIELD1_\$FIELD2_\$FIELD3_\$FIELD4_\$FIELD5.tiff

    if [ ! -f \$OUTPUT_FILE ]; then
        nextflow $cbs --py $py --x \$FIELD0 --y \$FIELD1 --size_x \$FIELD2 --size_y \$FIELD3 --ref_level \$FIELD4 --slide \$SLIDE --size \$SIZE -profile cluster
    else
        echo 'File exists, terminating process'
    fi
    """
}

