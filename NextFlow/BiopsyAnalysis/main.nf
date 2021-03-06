#!/usr/bin/env nextflow

/*  inputs */
params.in = "/share/data40T_v2/Peter/Data/Biopsy/"
TIFF_REMOTE = file(params.in + "*")
PublishPatient = "/share/data40T_v2/Peter/PatientFolder"
params.cleancore = file("/share/data40T_v2/Peter/.cleandir")

/* python files */
DistributedVersion = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/DistributedVersion.py')
WrittingTiff = file('/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/WrittingTiff/WriteFromFiles.py')

/* parameters */
WSI_MARGE = 50
MARGE_BIN = 1

process ChopPatient {
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    input:
    file PYTHONFILE from DistributedVersion
    file x from TIFF_REMOTE
    val wd_REMOTE from PublishPatient
    val marge from WSI_MARGE
    file cleandir from params.cleancore
    output:
    file "Job_${x.getBaseName()}" into JOB_SUBMIT
    file "Job_${x.getBaseName()}/ParameterDistribution.txt" into PARAM_JOB
    afterScript "bash $cleandir"
    """
    METHOD=grid_etienne
 
    python $PYTHONFILE --slide $x --output Job_${x.getBaseName()} --method \$METHOD --tc 10 --size_tiles 224 --marge $marge

    """
}

process SubImage {
//    executor 'sge'
    memory '16 GB'
//    profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash -q all.q -l mem_free=16G -pe orte 1 -R y"
    publishDir PublishPatient, overwrite: false
//    maxForks = 80
    errorStrategy 'retry' 
    maxErrors 50
    

    input:

    val p from PARAM_JOB.each().splitText() 
    val inputt from params.in
    val marge from MARGE_BIN
    val marge_wsi from WSI_MARGE
    file cleandir from params.cleancore
    output:
    file "Job_${p.split()[6]}/prob/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into PROB_PROCESSED
    file "Job_${p.split()[6]}/bin/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into BIN_PROCESSED
    file "Job_${p.split()[6]}/tiled/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into IMAGE_PROCESSED
    file "Job_${p.split()[6]}/table/${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.npy" into TABLE_PROCESSED, TABLE_PROCESSED2, TABLE_PROCESSED3
    afterScript "bash $cleandir"
    """


    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]}/PredictionSlide.py PredictionSlide.py
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${p.split()[6]} Job_${p.split()[6]}

    python PredictionSlide.py -x ${p.split()[1]} -y ${p.split()[2]} --size_x ${p.split()[3]} --size_y ${p.split()[4]} --ref_level ${p.split()[5]} --output Job_${p.split()[6]} --slide $inputt${p.split()[6]}.tiff --size 224 --marge $marge --marge_cut_off $marge_wsi

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
    val marge_wsi from WSI_MARGE
    output:
    file "Job_${key}/WSI/Segmented_${key}.tiff"

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${key} Job_${key}
    python $py $marge_wsi ${inputt}${key}.tiff ./Job_${key}/WSI/Segmented_${key}.tiff *.tiff 

    """
}





/* What is needed :
- tables (they will have nothing if no cells)
- origin slide to have dimensions
*/

/* python files */

MergeTable = file("MergeTabsAndPlot.py")

/* inputs */
TIFF_FOLDER = file(params.in)

TABLE_PROCESSED  .map { file -> tuple(getKey(file), file) }
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
    val inputt from params.in
    val marge_wsi from WSI_MARGE
    output:
    file "Job_${key}/${key}_whole_slide.csv" into TAB_SLIDE, TAB_SLIDE2
    file "Job_${key}/RankedTable/*.csv" into NEW_TAB mode flatten

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${key} Job_${key}
    python $py --slide ${inputt}${key}.tiff --marge_cut_off $marge_wsi
    """   

}



/* BEGIN: Creating distribution visualisation of features at the patient level */

DistributionPlot = file("DistributionPlot.py")

process FeatureDistribution {
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file wholeTab from TAB_SLIDE
    file py from DistributionPlot
    output:
    file "Job_${wholeTab.getBaseName().split('_')[0]}/Distribution/*.png" into histogramme
    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${wholeTab.name.split("_")[0]} Job_${wholeTab.name.split("_")[0]}
    python $py --table $wholeTab --output Job_${wholeTab.name.split("_")[0]}/Distribution 
    """


}

/* END: Creating distribution visualisation of features at the patient level */

/* BEGIN Creating feature map visualisation heatmap at resolution RES */

FeatureHeatMaps = file("FeatureHeatMaps.py")
SMOOTH = 5
RES = 5

process HeatMaps {
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file wholeTab from TAB_SLIDE2
    file py from FeatureHeatMaps
    val inputt from params.in
    val res from RES
    val smooth from SMOOTH
    output:
    file "Job_${wholeTab.getBaseName().split('_')[0]}/HeatMaps/*.png" into heatmaps
    beforeScript "source ~/.bashrc"
    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${wholeTab.name.split("_")[0]} Job_${wholeTab.name.split("_")[0]}
    python $py --table $wholeTab --output Job_${wholeTab.name.split("_")[0]}/HeatMaps --slide ${inputt}${wholeTab.name.split("_")[0]}.tiff --res $res --smooth $smooth
    """


}

/* END: Creating feature map visualisation */

/* BEGIN: Create colors maps at the WSI level */

/* file input */
ADDING_COLORS = file("AddingColors.py")

process MakeColors {
    clusterOptions = "-S /bin/bash"
    publishDir PublishPatient, overwrite: false
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file table from NEW_TAB
    file py from ADDING_COLORS
    val marge_wsi from WSI_MARGE
    output:
    file "Job_${table.getBaseName().split('_')[0]}/ColoredTiled/feat_*/${table.getBaseName()}.tiff" into COLOR_TIFF mode flatten

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${table.name.split("_")[0]} Job_${table.name.split("_")[0]}
    python $py --table $table --key ${table.name.split('_')[0]} --output Job_${table.name.split('_')[0]}/ColoredTiled --marge_cut_off $marge_wsi
    """   
}


// Here get parent folder to diff between patient and feature
def getColorKey( file ) {     
      file.toString().split('feat_')[1].split('/')[0] + "___" + file.toString().split('feat_')[1].split('/')[1].split('_')[0] 

}

COLOR_TIFF       .map { file -> tuple(getColorKey(file), file) }
                 .groupTuple() 
                 .set { COLOR_TIFF_GROUPED_BY_PATIENT_FEAT }

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
    val marge_wsi from WSI_MARGE
    output:
    file "Job_${key.split('___')[1]}/WSI/Segmented_${key}.tiff"

    """
    ln -s /share/data40T_v2/Peter/PatientFolder/Job_${key.split('___')[1]} Job_${key.split('___')[1]}
    python $py $marge_wsi ${inputt}${key.split('___')[1]}.tiff ./Job_${key.split('___')[1]}/WSI/Segmented_${key}.tiff *.tiff
    """
}
/* END: Create colors maps at the WSI level */
