
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
ADDING_COLORS = file("AddColors.py")
COLORING = file("Coloring.py")
COLORING2 = file("Coloring2.py")
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
    memory '11 GB'
    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 80
    errorStrategy 'retry' 
    maxErrors 5


    input:
    val p from ALL_CONFIG
    file param from PARAM_JOB.first()
    val inputt from params.in
    val marge from MARGE2.first()

    output:
    file "Job_${p.split()[6]}/prob/${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into PROB_PROCESSED
    file "Job_${p.split()[6]}/bin/${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into BIN_PROCESSED
    file "Job_${p.split()[6]}/tiled/${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into IMAGE_PROCESSED, IMAGE_PROCESSED2
    file "Job_${p.split()[6]}/table/${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.npy" into TABLE_PROCESSED, TABLE_PROCESSED2, TABLE_PROCESSED3
    """


    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]}/PredictionSlide.py PredictionSlide.py
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]} Job_${p.split()[6]}

    python PredictionSlide.py -x ${p.split()[1]} -y ${p.split()[2]} --size_x ${p.split()[3]} --size_y ${p.split()[4]} --ref_level ${p.split()[5]} --output Job_${p.split()[6]} --slide $inputt${p.split()[6]}.tiff --size 224 --marge $marge

    """


}
/*
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
//METRICS.subscribe { println "value: $it" }
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
    file metrics from METRICS.toList()
    file py from ASSEMBLE
    output:
    file "METRIC_GENERAL_*.npy" into METRIC_GEN, METRIC_GEN2

    """
    python $py --path .
    """
}

process MakeColors {
    executor 'sge'
    profile = 'cluster'
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file metrics from METRIC_GEN
    each table from TABLE_PROCESSED2
    file py from ADDING_COLORS
    output:
    file "*_color.npy" into TABLE_COLORED, TABLE_COLORED2

    """
    python $py --table $table
    """   
}
*/
RES = 7
GOING_TO_RES = file("ChangingRes.py")

process MergeTables {
    executor 'sge'
    profile = 'cluster'
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file table from TABLE_PROCESSED3
    file py from GOING_TO_RES.last()
    val res from RES
    output:
    file "*_general.csv" into TABLES

    """
    python $py --table $table --resolution $res
    """   
}

MergeTable = file("MergeTabsAndPlot.py")

process CollectMergeTables {
    executor 'local'
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file _ from TABLES .toList()
    file py from MergeTable
    val res from RES
    output:
    file "*_WHOLE.csv" into TAB_SLIDE

    """
    python $py --resolution $res
    """   

}



/*
//TABLE_COLORED.combine(BIN_PROCESSED).
//test = Channel.from(TABLE_COLORED)
//test.subscribe { println "value: $it" }

//println(TABLE_COLORED.getClass())
//TABLE_COLORED.combine(bin).filter { it[1] == it[2]}

// FIND BETTER WAY OF DOING IT
process AddColors {
    executor 'sge'
    profile = 'cluster'
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file t_color from TABLE_COLORED
    file py from COLORING
    file py2 from COLORING2
//    file bin from BIN_PROCESSED
    """
    python $py
    python $py2
    """

}


*/
