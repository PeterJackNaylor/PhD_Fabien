
/*  inputs */
params.in = "/share/data40T_v2/Peter/Data/Biopsy/"
TIFF_REMOTE = file(params.in + "*")
PublishPatient = "/share/data40T_v2/Peter/PatientFolder"

/* python files */
DistributedVersion = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')


/* parameters */
MARGE = 100


process ChopPatient {
    executor 'sge'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false

    input:
    file PYTHONFILE from DistributedVersion
    file x from TIFF_REMOTE
    val wd_REMOTE from PublishPatient
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

process SubImage {
    executor 'sge'
    memory '11 GB'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    maxForks = 80
    errorStrategy 'retry' 
    maxErrors 50


    input:
    val p from PARAM_JOB.each().splitText() 
    val inputt from params.in
    val marge from MARGE2.first()

    output:
    file "Job_${p.split()[6]}/prob/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into PROB_PROCESSED
    file "Job_${p.split()[6]}/bin/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into BIN_PROCESSED
    file "Job_${p.split()[6]}/tiled/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into IMAGE_PROCESSED, IMAGE_PROCESSED2
    file "Job_${p.split()[6]}/table/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.npy" into TABLE_PROCESSED, TABLE_PROCESSED2
    """


    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]}/PredictionSlide.py PredictionSlide.py
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]} Job_${p.split()[6]}

    python PredictionSlide.py -x ${p.split()[1]} -y ${p.split()[2]} --size_x ${p.split()[3]} --size_y ${p.split()[4]} --ref_level ${p.split()[5]} --output Job_${p.split()[6]} --slide $inputt${p.split()[6]}.tiff --size 224 --marge $marge

    """
}


/* this process creates files {slide}_{x}_{y}_{w}_{h}_{r}.tiff */
/* Creating feature map visualisation */

/* What is needed :
- tables (they will have nothing if no cells)
- origin slide to have dimensions
*/

/* python files */

MergeTable = file("MergeTabsAndPlot.py")
ChangingRes = file("ChangingRes.py")

/* inputs */
TIFF_FOLDER = file(params.in)
RES = 7

TABLE_PROCESSED.each() {print it.getText()}
/*
process MergeTablesBySlides {
    executor 'sge'
    profile = 'cluster'
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file table from TABLE_PROCESSED
    file py from ChangingRes.last()
    val res from RES
    file fold from TIFF_FOLDER
    output:
    file "Job_${table.getText().split("_")[0]}/tables_res_0/${table.getText().split("_")[0]}_tables_res_0.csv" into Tables_res_0

    """

    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${table.split("_")[0]} Job_${table.split("_")[0]}

    python $py --table $table --slide ${fold}/${table.split("_")[0]}.tiff --resolution $res
    """   
}
*/
/*
process CollectMergeTables {
    executor 'local'
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file _ from Tables_res_0 .toList()
    file py from MergeTable
    val res from RES
    output:
    file "*_WHOLE.csv" into TAB_SLIDE

    """
    python $py --resolution $res
    """   

}
*/
/* END: Creating feature map visualisation */
