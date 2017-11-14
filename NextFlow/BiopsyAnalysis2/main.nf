#!/usr/bin/env nextflow

/*  inputs */
params.in = "/share/data40T_v2/Peter/Data/Biopsy"
params.python = "/share/data40T_v2/Peter/PythonScripts/PhD_Fabien"
params.publish = "/share/data40T_v2/Peter/PatientFolder"
params.cleancore = file("/share/data40T_v2/Peter/.cleandir")
params.pretrained = file("/share/data40T_v2/Peter/pretrained_models")
params.home = "/share/data40T_v2/Peter"

CHOP = file('src/Chop.py')
WSI_MARGE = 50
LAMBDA = 7
SMALLOBJECT = 50
SMOOTH = 5
RES = 5
TIFF_REMOTE = file(params.in + "/*.tiff")
WrittingTiff = file(params.python + '/WrittingTiff/WriteFromFiles.py')
PREDICT = file("src/Predict.py")
PREDICTGPU = file("src/PredictGPU.py")
EXTRACTOR = file('src/Extraction.py')
MergeTable = file("src/MergeTabsAndPlot.py")
DistributionPlot = file("src/DistributionPlot.py")
FeatureHeatMaps = file("src/FeatureHeatMaps.py")
ADDING_COLORS = file("src/AddingColors.py")


process ChopPatient {
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash -q all.q"
//    publishDir PublishPatient, overwrite: false
    input:
    file PYTHONFILE from CHOP
    file x from TIFF_REMOTE
    val marge from WSI_MARGE
    file cleandir from params.cleancore
    output:
    file "Parameter.txt" into PARAM_JOB
    file "$x" into SLIDES
    afterScript "bash $cleandir"
    """
    METHOD=grid_etienne
    python $PYTHONFILE --slide $x --output Parameter.txt --method \$METHOD --marge $marge
    """
}

process ProbabilityMap {
    memory '16 GB'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash -q cuda.q"
    maxForks = 1
//    errorStrategy 'retry' 
//    maxErrors 50
    

    input:
    val home from params.home
    file pred from PREDICTGPU
    file p from PARAM_JOB 
    file slide from SLIDES
    file cleandir from params.cleancore
    val train_folder from params.pretrained
    output:
    file "prob_*.tiff" into PROB, PROB2 mode flatten
    file "rgb_*.tiff" into RGB, RGB2  mode flatten

    beforeScript "source ${home}/CUDA_LOCK/.whichNODE"
    afterScript "source ${home}/CUDA_LOCK/.freeNODE"
    afterScript "bash $cleandir"
    """
    python $pred --slide $slide --parameter $p --output . --trained $train_folder
    """
}

/*
process ProbabilityMap {
    memory '16 GB'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash -q all.q -l mem_free=16G -pe orte 6 -R y"
//    cpu 2
//    publishDir PublishPatient, overwrite: false
    maxForks = 50
    errorStrategy 'retry' 
    maxErrors 50
    

    input:
    file pred from PREDICT
    val p from PARAM_JOB.each().splitText() 
    val inputt from params.in
    file cleandir from params.cleancore
    val train_folder from params.pretrained
    output:
    file "prob_${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into PROB, PROB2
    file "rgb_${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff" into RGB, RGB2
    afterScript "bash $cleandir"
    """
    python $pred -x ${p.split()[1]} -y ${p.split()[2]} --size_x ${p.split()[3]} --size_y ${p.split()[4]} --ref_level ${p.split()[5]} --slide $inputt/${p.split()[6]}.tiff --output prob_${p.split()[6]}_${p.split()[1]}_${p.split()[2]}_${p.split()[3]}_${p.split()[4]}_${p.split()[5]}.tiff --trained $train_folder
    """
}*/


process BinaryMaps {
    clusterOptions = "-S /bin/bash -q all.q -pe orte 1 -R y"
    errorStrategy 'retry' 
    maxErrors 50
    input:
    file probs from PROB
    val param from LAMBDA
    val smallobject from SMALLOBJECT
    output:
    file "bin_*.tiff" into BIN, BIN2
    """
    #!/usr/bin/env python
    from tifffile import imread, imsave
    from skimage.morphology import remove_small_objects
    from Deprocessing.Morphology import DynamicWatershedAlias
    import numpy as np

    probs = imread("${probs}")
    probs = probs.astype(float)
    probs = probs / 255


    bin_img = DynamicWatershedAlias(probs, ${param})
    bin_img = remove_small_objects(bin_img, ${smallobject})

    bin_img[bin_img > 0] = 255
    bin_img = bin_img.astype(np.uint8)
    imsave('${probs}'.replace('prob', 'bin'), bin_img, resolution=[1.0,1.0])
    """
}

def getPositionID( file ) {
        file.baseName.split('_') .drop(1).join('_')
}
RGB .phase(BIN, remainder: true) { it -> getPositionID(it) } .set { RGB_AND_BIN }


process ExtractionFeatures {
    clusterOptions = "-S /bin/bash -q all.q"
    input:
    set file(rgb), file(bin) from RGB_AND_BIN
    val py from EXTRACTOR
    val marge from WSI_MARGE
    output:
    file "segmented_*.tiff" into SEG
    file "table_*.csv" into TABLE
    """
    python $py --bin $bin --rgb $rgb --marge $marge
    """
}

def getKey( file ) {
      file.name.split('_')[1] 
}

TABLE  .map { file -> tuple(getKey(file), file) }
                 .groupTuple() 
                 .set { TableGroups }

process CollectMergeTables {
    executor 'local'
    maxForks = 200
    errorStrategy 'retry' 
    maxErrors 5
    publishDir "./tables", overwrite: false, pattern: "${key}.csv"


    input:
    set key, file(tables) from TableGroups 
    file py from MergeTable
    val inputt from params.in
    val marge_wsi from WSI_MARGE
    output:
    file "${key}.csv" into TAB_WSI, TAB_WSI2, TAB_WSI3
    file "Ranked_${key}_*/*.csv" into NEW_TAB mode flatten
    """
    python $py --slide ${inputt}/${key}.tiff --marge $marge_wsi
    """   
}

SEG  .map { file -> tuple(getKey(file), file) }
                 .groupTuple() 
                 .set { SegmentedByPatient }

process SegStiching {
    clusterOptions = "-S /bin/bash -q all.q -l mem_free=16G"
    publishDir "./Segmented", overwrite: false
    errorStrategy 'retry' 
    maxErrors 50
    input:
    set key, file(vec_color) from SegmentedByPatient
    val inputt from params.in
    file py from WrittingTiff
    val marge_wsi from WSI_MARGE
    output:
    file "Segmented_${key}.tiff"
    """
    python $py $marge_wsi ${inputt}/${key}.tiff Segmented_${key}.tiff *.tiff 
    """
}

process FeatureDistribution {
    clusterOptions = "-S /bin/bash -q all.q"
    publishDir "./FeatureDistribution", overwrite: false
    errorStrategy 'retry' 
    maxErrors 5
    input:
    file wholeTab from TAB_WSI
    file py from DistributionPlot
    output:
    file "${wholeTab.baseName}" into histogramme
    """
    python $py  --table ${wholeTab} --output ${wholeTab.baseName}
    """
}

process HeatMaps {
    clusterOptions = "-S /bin/bash -q all.q"
    publishDir "./Heatmaps", overwrite: false
    errorStrategy 'retry' 
    maxErrors 5

    input:
    file wholeTab from TAB_WSI2
    file py from FeatureHeatMaps
    val inputt from params.in
    val res from RES
    val smooth from SMOOTH
    output:
    file "${wholeTab.baseName}/*.png" into heatmaps
    beforeScript "source ~/.bashrc"
    """
    python $py --table $wholeTab --output ${wholeTab.baseName} --slide ${inputt}/${wholeTab.baseName}.tiff --res $res --smooth $smooth
    """
}
def getPositionIDWithoutEnd( file ) {
      file.baseName.split('_').drop(1).join('_')
}
NEW_TAB .phase(BIN2, remainder: true) { it -> getPositionIDWithoutEnd(it) } .set { TAB_AND_BIN }

process MakeColors {
    clusterOptions = "-S /bin/bash -q all.q"
    errorStrategy 'retry' 
    maxErrors 5

    input:
    set file(table), file(bin) from TAB_AND_BIN
    file py from ADDING_COLORS
    val marge_wsi from WSI_MARGE
    file wholeTab from TAB_WSI3 .toList()
    output:
    file "feat_*/${table.getBaseName()}.tiff" into COLOR_TIFF mode flatten

    """
    python $py --table $table --key ${table.name.split('_')[1]} --output . --marge $marge_wsi --bin $bin
    """   
}

def getColorKey( file ) {     
      file.toString().split('feat_')[1].split('/')[0] + "___" + file.toString().split('feat_')[1].split('/')[1].split('_')[1] 

}

COLOR_TIFF       .map { file -> tuple(getColorKey(file), file) }
                 .groupTuple() 
                 .set { COLOR_TIFF_GROUPED_BY_PATIENT_FEAT }
process StichingFeatTiff {
    memory '11 GB'
    clusterOptions = "-S /bin/bash"
    publishDir "./FeatureWSI/${key.split('___')[1]}", overwrite: false
    errorStrategy 'retry' 
    maxErrors 50
    input:
    set key, file(tiled_color) from COLOR_TIFF_GROUPED_BY_PATIENT_FEAT
    val inputt from params.in
    file py from WrittingTiff
    val marge_wsi from WSI_MARGE
    output:
    file "${key}.tiff"

    """
    python $py $marge_wsi ${inputt}/${key.split('___')[1]}.tiff ./${key}.tiff *.tiff
    """
}

