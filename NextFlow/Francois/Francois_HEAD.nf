#!/usr/bin/env nextflow


TOANNOTATE = "/share/data40T_v2/Peter/Data/ToAnnotate"
wd_datagen = "/share/data40T_v2/Peter/Francois"

PY = file("first.py")
PYextract = file("second.py")


CHANGEENV = file('ChangeEnv.py')
params.in = file("/share/data40T_v2/Peter/Francois/New_images_TMA_ICA/*") 
params.out = file("/share/data40T_v2/Peter/Francois/OutFixing")



process ChangeEnv {

    executor 'sge'
    profile = 'cluster'

    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"

    input:
    val env from TOANNOTATE
    val wd from wd_datagen
    file py from CHANGEENV

    output:
    val wd into WD
    """
    python $py --env $env --wd $wd
    """

}

process ProcessPatient {

    executor 'sge'
    profile = 'cluster'
    queue = "all.q"
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"
    memory = '15G'
    publishDir params.out, mode: "copy", overwrite: false
    maxForks = 20
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file image from params.in
    val wd from WD .first()
    file py from PY
    val env from TOANNOTATE

    output:
    file "*_40x" into raw // or something similar to get the folder
       
    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    """

    python $py --output . --file_name $image --env $env --wd $wd
    """
}

process ExtractFeatures {

    executor 'sge'
    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir params.out, mode: "copy", overwrite: false
    maxForks = 50
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file folder from raw
    file py from PYextract

    output:
    file "$folder/${folder}_table.csv" // or numpy check groovy

    """
    python $py --folder_name $folder
    """

}
