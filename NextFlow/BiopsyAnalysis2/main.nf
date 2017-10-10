#!/usr/bin/env nextflow

/*  inputs */
params.in = "/share/data40T_v2/Peter/Data/Biopsy"
params.python = "/share/data40T_v2/Peter/PythonScripts/PhD_Fabien"
params.publish = "/share/data40T_v2/Peter/PatientFolder"
params.cleancore = file("/share/data40T_v2/Peter/.cleandir")

CHOP = file('Chop.py')
WSI_MARGE = 50
TIFF_REMOTE = file(params.in + "/*.tiff")

process ChopPatient {
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"
//    publishDir PublishPatient, overwrite: false
    input:
    file PYTHONFILE from CHOP
    file x from TIFF_REMOTE
    val marge from WSI_MARGE
    file cleandir from params.cleancore
    output:
    file "Parameter.txt" into PARAM_JOB
    afterScript "bash $cleandir"
    """
    METHOD=grid_etienne
 
    python $PYTHONFILE --slide $x --output Parameter.txt --method \$METHOD --marge $marge
    """
}


PREDICT = file("Predict.py")
params.pretrained = "/share/data40T_v2/Peter/pretrained_models"

process SubImage {
//    executor 'sge'
    memory '16 GB'
//    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash -q all.q -l mem_free=16G -pe orte 1 -R y"
//    publishDir PublishPatient, overwrite: false
//    maxForks = 80
    errorStrategy 'retry' 
    maxErrors 50
    

    input:
    file pred from PREDICT
    val p from PARAM_JOB.each().splitText() 
    val inputt from params.in
    file cleandir from params.cleancore
    val train_folder from params.pretrained
    output:
    file "prob_${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into PROB
    afterScript "bash $cleandir"
    """

    python $pred -x ${p.split()[1]} -y ${p.split()[2]} --size_x ${p.split()[3]} --size_y ${p.split()[4]} --ref_level ${p.split()[5]} --slide $inputt/${p.split()[6]}.tiff --output prob_${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff --trained $train_folder

    """
}