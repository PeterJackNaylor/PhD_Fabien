#!/usr/bin/env nextflow

params.python_dir = '/share/data40T_v2/Peter/PythonScripts/PhD_Fabien'
params.toannotate = "/share/data40T_v2/Peter/Data/ToAnnotate"
params.net = '/share/data40T_v2/Peter/Francois'
params.neeraj = "/share/data40T_v2/Peter/Data/NeerajKumar/ForDatagen"
params.cleancore = file("/share/data40T_v2/Peter/.cleandir")
PROCESS = file('ApplyPostProcess.py')
DataNeeraj = file(params.neeraj + "/*.png")
IMAGE = file(params.neeraj + "/Slide_Breast/*.png")
GT = file(params.neeraj + "/GT_Breast/*.png")
STEPSIZE = [50, 75, 100, 125, 150, 175, 200, 224]
LAMBDA = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
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
    queue = "all.q"
    input:
    file py from PROCESS .last()
    val image from IMAGE
    val gt from GT
    each stepSize from STEPSIZE 
    each clearborder from CLEARBORDER
    each lambda from LAMBDA
    val wd from WD .first()
    file cleandir from params.cleancore
    output:
    file "${clearborder}__${stepSize}__max__${lambda}.csv" into result
    afterScript "bash $cleandir"
    """
    python $py --wd $wd --image $image --gt $gt --stepsize $stepSize --method max --clearborder ${clearborder} --lambda $lambda --output ${clearborder}__${stepSize}__max__${lambda}.csv
    """
}

process Best_Stiching_others {
    queue = "all.q"
    executor 'sge'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"

    input:
    file py from PROCESS .last()
    val image from IMAGE
    val gt from GT
    each stepSize from STEPSIZE 
    each clearborder from CLEARBORDER2
    each method from METHOD
    each lambda from LAMBDA
    val wd from WD2 .first()
    file cleandir from params.cleancore
    output:
    file "${clearborder}__${stepSize}__${method}__${lambda}.csv" into result2
    afterScript "bash $cleandir"

    """
    python $py --wd $wd --image $image --gt $gt --stepsize $stepSize --method $method --clearborder $clearborder --lambda $lambda --output ${clearborder}__${stepSize}__${method}__${lambda}.csv
    """
}

process RegroupResults {

    clusterOptions = "-S /bin/bash"
    publishDir "./Results", overwrite: true

    input:
    file fold from result .toList()
    file fold2 from result2 .toList()
    output:
    file "results.csv" into RES

    """
    #!/usr/bin/env python
    import os
    import pandas as pd
    import pdb
    from glob import glob
    CSV = glob('*.csv')
    df_list = []
    for f in CSV:
        df = pd.read_csv(f, index_col=0)
        df_list.append(df)
    table = pd.concat(df_list)
    table.to_csv('results.csv')

    """

}


BARCHARTS = file("StichBarCharts.py")

process PlotResults {

    clusterOptions = "-S /bin/bash"
    publishDir "./Results", overwrite: true

    input:
    file res from RES
    file py from BARCHARTS
    output:
    file "*.png"
    
    """
    python $py
    """
}
