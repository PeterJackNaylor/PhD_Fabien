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
WD_REMOTE = "/share/data40T_v2/Peter/PatientFolder"

DATA_FOLDER = "Data"

DISTRIBUTED_VERSION = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')
GETMAX = file('GetMax.py')
ASSEMBLE = file("Assemble.py")
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
    memory '9 GB'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 40
    errorStrategy 'retry' 
    maxErrors 5


    input:
    val p from ALL_CONFIG
    file param from PARAM_JOB.first()
    val inputt from params.in
    val marge from MARGE2.first()

    output:
    file "Job_${p.split()[6]}/tiled/${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into IMAGE_PROCESSED, IMAGE_PROCESSED2
    file "Job_${p.split()[6]}/table/${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.npy" into TABLE_PROCESSED
    """


    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]}/PredictionSlide.py PredictionSlide.py
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]} Job_${p.split()[6]}

    python PredictionSlide.py -x ${p.split()[1]} -y ${p.split()[2]} --size_x ${p.split()[3]} --size_y ${p.split()[4]} --ref_level ${p.split()[5]} --output Job_${p.split()[6]} --slide $inputt${p.split()[6]}.tiff --size 224 --marge $marge

    """


}

//TABLE_PROCESSED.subscribe { println "value: $it" }
FEATURES_TO_VISU = [0, 1, 2]
process GetMax {
    executor 'sge'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 200
//    errorStrategy 'retry' 
    maxErrors 5

    input:
    file table from TABLE_PROCESSED 
    file py from GETMAX
    each feat from FEATURES_TO_VISU 
    output:
    file "*.npy" into METRICS

    """
    python $py --table $table --feat $feat
    """
    
}
/*
process BringToGether {
    executor 'sge'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 20
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file metrics from METRICS
    file py from ASSEMBLE
    output:
    file "METRIC_GENERAL.npy" into METRIC_GEN

    """
    python $py --path .
    """


}*/


