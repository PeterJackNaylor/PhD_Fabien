#!/usr/bin/env nextflow

params.python_dir = '/share/data40T_v2/Peter/PythonScripts/PhD_Fabien'
params.toannotate = "/share/data40T_v2/Peter/Data/ToAnnotate"
params.net = '/share/data40T_v2/Peter/Francois'
params.neeraj = "/share/data40T_v2/Peter/Data/NeerajKumar/ForDatagen"
PROCESS = file('ApplyPostProcess.py')
DataNeeraj = file(params.neeraj + "/*.png")
IMAGE = file(params.neeraj + "/Slide_Breast/*.png")
GT = file(params.neeraj + "/GT_Breast/*.png")
STEPSIZE = [50, 75, 100, 125, 150, 175, 200, 224]
LAMBDA = [5, 6, 7, 8, 9, 10, 11, 12, 13]
CLEARBORDER = ["RemoveBorderObjects", "RemoveBorderWithDWS", "Reconstruction", "Classic"]
CLEARBORDER2 = ["RemoveBorderObjects", "RemoveBorderWithDWS", "Classic"]
METHOD = ["avg", "median"]
CHANGEENV = file(params.python_dir + '/NextFlow/Francois/ChangeEnv.py')

process ChangeEnv {

    executor 'sge'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"

    input:
    val env from params.toannotate
    val wd from params.net
    file py from CHANGEENV

    output:
    val wd into WD, WD2
    """
    python $py --env $env --wd $wd
    """

}

process Best_Stiching {

    executor 'sge'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"

    input:
    file py from PROCESS
    val image from IMAGE
    val gt from GT
    each stepSize from STEPSIZE 
    each clearborder from CLEARBORDER
    each lambda from LAMBDA
    val wd from WD
    output:
    file '${clearborder}__${stepSize}__max_${lambda}.csv' into result

    """
    python $py --wd $wd --image $image --gt $gt --stepsize $stepSize --method max --clearborder ${clearborder} --lambda $lambda --output ${clearborder}__${stepSize}__max__${lambda}.csv
    """
}

process Best_Stiching_others {

    executor 'sge'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"

    input:
    file py from Process
    file image from IMAGE
    file gt from GT
    each stepSize from STEPSIZE 
    each clearborder from CLEARBORDER2
    file method from METHOD
    each lambda from LAMBDA
    val wd from WD2
    output:
    file "${clearborder}__${stepSize}__${method}__${lambda}.csv" into result2

    """
    python $py --wd $wd --image $image --gt $gt --stepsize $stepSize --method $method --clearborder $clearborder --lambda $lambda --output ${clearborder}__${stepSize}__${method}__${lambda}.csv
    """
}



