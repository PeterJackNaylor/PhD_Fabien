#!/usr/bin/env nextflow

/*  inputs */
params.in = "/share/data40T_v2/Peter/Data/Biopsy/"
TIFF_REMOTE = file(params.in + "*")
PublishPatient = "/share/data40T_v2/Peter/PatientFolder"

/* python files */
DistributedVersion = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')
WrittingTiff = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/WriteFromFiles.py')

/* parameters */
MARGE = 100


process ChopPatient {
//    executor 'sge'
//    profile = 'cluster'
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
//    executor 'sge'
    memory '11 GB'
//    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
//    maxForks = 80
    errorStrategy 'retry' 
    maxErrors 50


    input:

    val p from PARAM_JOB.each().splitText() 
    val inputt from params.in
    val marge from MARGE2.first()

    output:
    file "Job_${p.split()[6]}/prob/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into PROB_PROCESSED
    file "Job_${p.split()[6]}/bin/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into BIN_PROCESSED
    file "Job_${p.split()[6]}/tiled/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into IMAGE_PROCESSED
    file "Job_${p.split()[6]}/table/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.npy" into TABLE_PROCESSED, TABLE_PROCESSED2, TABLE_PROCESSED3
    """


    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]}/PredictionSlide.py PredictionSlide.py
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]} Job_${p.split()[6]}

    python PredictionSlide.py -x ${p.split()[1]} -y ${p.split()[2]} --size_x ${p.split()[3]} --size_y ${p.split()[4]} --ref_level ${p.split()[5]} --output Job_${p.split()[6]} --slide $inputt${p.split()[6]}.tiff --size 224 --marge $marge

    """
}

/* this process creates files {slide}_{x}_{y}_{w}_{h}_{r}.tiff */



def getKey( file ) {
      file.name.split('_')[0] 
}
/* Regrouping files by patient for tiff stiching */

IMAGE_PROCESSED  .map { file -> tuple(getKey(file), file) }
                 .groupTuple() 
                 .set { SegmentedByPatient }

process StichingTiff {
    memory '11 GB'
//    profile = 'cluster'
    validExitStatus 0
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
//    maxForks = 80
    errorStrategy 'retry' 
    maxErrors 50
    input:
    set key, file(vec_color) from SegmentedByPatient
    val inputt from params.in
    file py from WrittingTiff
    output:
    file "Job_${key}/WSI/Segmented_${key}.tiff"

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${key} Job_${key}
    python $py ${inputt}${key}.tiff ./Job_${key}/WSI/Segmented_${key}.tiff *.tiff

    """


}



/* Creating feature map visualisation heatmap at resolution RES */


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
    file py from ChangingRes
    val res from RES
    file fold from TIFF_FOLDER
    output:

    file "Job_${table.getBaseName().split('_')[0]}/tables_res_0/${table.getBaseName()}_tables_res_0.csv" into Tables_res_0

    """

    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${table.name.split("_")[0]} Job_${table.name.split("_")[0]}

    python $py --table $table --slide ${fold}/${table.name.split("_")[0]}.tiff --resolution $res
    """   
}


Tables_res_0     .map { file -> tuple(getKey(file), file) }
 		         .groupTuple() 
     		     .set { TableGroups }

process CollectMergeTables {
    executor 'local'
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5

    input:
    set key, file(tables) from TableGroups 
    file py from MergeTable
    val res from RES
    val inputt from params.in
    output:
    file "Job_${key}/${key}_whole_slide.csv" into TAB_SLIDE

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${key} Job_${key}
    python $py --resolution $res --slide ${inputt}${key}.tiff
    """   

}

/* END: Creating feature map visualisation */

/* BEGIN: Create colors maps at the WSI level */

/* What is needed :
- tables (they will have nothing if no cells)
- origin slide to have dimensions
*/

/* python files */

GETSTATISTICS_4_COLORS = file("GetStatistics4Color.py")

/* inputs */
FEATURES_TO_VISU = [0, 1, 2]

/* Resume each table */
process GetStatistics4Color {
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file table from TABLE_PROCESSED2
    file py from GETSTATISTICS_4_COLORS
    each feat from FEATURES_TO_VISU 
    output:
    file "Job_${table.getBaseName().split('_')[0]}/StatColors/${table.getBaseName()}_${feat}_color_0.npy" into COLOR_VEC

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${table.name.split("_")[0]} Job_${table.name.split("_")[0]}

    python $py --table $table --feat $feat --output Job_${table.name.split('_')[0]}/StatColors
    """
}


COLOR_VEC     .map { file -> tuple(getKey(file), file) }
                 .groupTuple() 
                 .set { COLOR_VEC_BY_PATIENT }


MergeStatsByWSI = file("GeneralStatistics4Color.py")



process BringTogetherStatistics4Color {
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    errorStrategy 'retry' 
    maxErrors 5

    input:
    set key, file(vec_color) from COLOR_VEC_BY_PATIENT
    file py from MergeStatsByWSI
    output:
    file "Job_${key}/GeneralStats4Color/GeneralStatistics4color_${key}_*.npy" into GeneralStatsByPatientByFeat


    """
    python $py --path . --output Job_${key}/GeneralStats4Color --key ${key}
    """
}
/* file input */
ADDING_COLORS = file("AddingColors.py")


process MakeColors {
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file table from TABLE_PROCESSED3
    file py from ADDING_COLORS
    output:
    file "Job_${table.getBaseName().split('_')[0]}/ColoredTiled/feat_*/${table.getBaseName()}.tiff" into COLOR_TIFF mode flatten

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${table.name.split("_")[0]} Job_${table.name.split("_")[0]}
    python $py --table $table --key ${table.name.split('_')[0]} --output Job_${table.name.split('_')[0]}/ColoredTiled
    """   
}


// Here get parent folder to diff between patient and feature
def getColorKey( file ) {     
      file.toString().split('feat_')[1].split('/')[0] + "___" + file.toString().split('feat_')[1].split('/')[1].split('_')[0] 

}

COLOR_TIFF       .map { file -> tuple(getColorKey(file), file) }
                 .groupTuple() 
                 .set { COLOR_TIFF_GROUPED_BY_PATIENT_FEAT }

// COLOR_TIFF_GROUPED_BY_PATIENT_FEAT .subscribe { println "value: ${it[0]}" }
process StichingFeatTiff {
    memory '11 GB'
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
//    maxForks = 80
    errorStrategy 'retry' 
    maxErrors 50
    input:
    set key, file(tiled_color) from COLOR_TIFF_GROUPED_BY_PATIENT_FEAT
    val inputt from params.in
    file py from WrittingTiff
    output:
    file "Job_${key.split('___')[1]}/WSI/Segmented_${key}.tiff"

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${key.split('___')[1]} Job_${key.split('___')[1]}
    python $py ${inputt}${key.split('___')[1]}.tiff ./Job_${key.split('___')[1]}/WSI/Segmented_${key}.tiff *.tiff"""


}
