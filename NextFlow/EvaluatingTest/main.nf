#!/usr/bin/env nextflow

// nextflow main.nf -profile mines --image_dir /share/data40T_v2/Peter/Data --python_dir /share/data40T_v2/Peter/PythonScripts/PhD_Fabien --home /share/data40T_v2/Peter -resume

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
TENSORBOARD = file(params.image_dir + '/UNet3')
MEANPY = file(params.python_dir + '/Data/MeanCalculation.py')

EXP = params.home + '/TF_EXP'

DataNeeraj = file(params.image_dir + "/NeerajKumar/ForDatagen")

ORGANS = ["Breast", "Bladder"]


process Mean {

    clusterOptions = "-S /bin/bash"
    queue 'all.q'    

    input:
    file py from MEANPY
    file toannotate from IMAGE_FOLD
    output:
    file "mean_file.npy" into MeanFile

    """
    python $py --path $toannotate --output .
    """
}


process PrepareImagesUNet {

    clusterOptions = "-S /bin/bash"
    queue 'all.q'

    input:
    file input from DataNeeraj
    val organs from ORGANS

    output:
    file "Slide___*.png" into SLIDE_UNET mode flatten
    file "GT___*.png" into GT_UNET mode flatten
    """
    #!/usr/bin/env python

    from UsefulFunctions.ImageTransf import ListTransform
    from Data.DataGenClass import DataGenMulti
    from scipy.misc import imsave
    _, transform_list_test = ListTransform()
    DG = DataGenMulti("$input", split='test', crop = 1, size=(1000, 1000),seed_=42,
                      transforms=transform_list_test, UNet=True, num="$organs")
    key = 0
    for _ in range(DG.length):
        key = DG.NextKeyRandList(key)
        img, anno = DG[key]
        name = "Slide___{}_{}.png".format("$organs", _)
        gt = "GT___{}_{}.png".format("$organs", _)
        imsave(name, img)
        imsave(gt, anno)
    """
}

UNET2_EXP = [EXP + "/classic/32_0.00005_0.0001", EXP + "/no_hsv_he_elast/32_0.00005_0.001", EXP + "/no_elast/32_0.0005_0.0001", EXP + "/no_he/32_0.00005_0.001", EXP + "/no_hsv/32_0.00005_0.001", EXP + "/no_hsv_he/32_0.00005_0.001", EXP + "/nothing/32_0.00005_0.001"]
PARAM = 10
UNETPREDICTION = file('UNetPrediction.py')

process UNetBN_2 {

    queue "all.q"
    clusterOptions = "-S /bin/bash"

    input:
    file py from UNETPREDICTION .last()
    file image from SLIDE_UNET
    file anno from GT_UNET
    file mean_file from MeanFile .first()
    val param from PARAM
    each fold from UNET2_EXP
    output:
    file "${image.getBaseName()}_*-_*" into AnalyseFolder

    """
    PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $py -i $image -a $anno -f $fold --mean_file $mean_file --param $param 
    """


}


process PrepareImages {
    clusterOptions = "-S /bin/bash"
    queue 'all.q'
    input:
    file input from DataNeeraj
    val organs from ORGANS

    output:
    file "Slide___*.png" into SLIDE mode flatten
    file "GT___*.png" into GT mode flatten
    """
    #!/usr/bin/env python

    from UsefulFunctions.ImageTransf import ListTransform
    from Data.DataGenClass import DataGenMulti
    from scipy.misc import imsave
    _, transform_list_test = ListTransform()
    DG = DataGenMulti("$input", split='test', crop = 1, size=(1000, 1000), seed_=42,
                      transforms=transform_list_test, UNet=False, num="$organs")
    key = 0
    for _ in range(DG.length):
        key = DG.NextKeyRandList(key)
        img, anno = DG[key]
        name = "Slide___{}_{}.png".format("$organs", _)
        gt = "GT___{}_{}.png".format("$organs", _)
        imsave(name, img)
        imsave(gt, anno)
    """
}
wd_datagen = params.home + "/Francois"
 
TOANNOTATE = params.image_dir + "/ToAnnotate"
PREDICTING_CAFFE = file("PredictingCaffe.py")
CHANGEENV = file(params.python_dir + '/NextFlow/Francois/ChangeEnv.py')
PARAM = 8
STEPSIZE = 150
NET1="DeconvNet_0.01_0.99_0.0005"
NET2="FCN_0.01_0.99_0.005"

process ChangeEnv {

    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"
    queue 'all.q'

    input:
    val env from TOANNOTATE
    val wd from wd_datagen
    file py from CHANGEENV

    output:
    val wd into WD
    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    """
    python $py --env $env --wd $wd
    """

}

process Ensemble {

    queue 'all.q'
    clusterOptions = "-S /bin/bash"
    validExitStatus 0,134 

    input:
    file py from PREDICTING_CAFFE .last()
    file image from SLIDE
    file anno from GT
    val param from PARAM
    val stepsize from STEPSIZE
    val net1 from NET1
    val net2 from NET2
    val wd from WD .first()
    file env from TOANNOTATE
    output:
    file "${image.getBaseName()}_*-_*" into AnalyseFolderCaffe mode flatten
    beforeScript 'export PYTHONPATH=/cbio/donnees/pnaylor/PythonPKG/caffe_peter2_cpu/python:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/FCN_Segmentation:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/UsefulFunctions:/share/data40T_v2/Peter/PythonScripts/PhD_Fabien/Nets:/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster:/share/data40T/pnaylor/Cam16/scripts/challengecam/PythonPatch:/share/data40T/pnaylor/Cam16/scripts/challengecam/RandomForest_Peter:/share/apps/user_apps/smil_0.8.1/lib/Smil/'

    """
    python $py --output . --env $env --wd $wd --image $image --anno $anno --param $param --stepSize $stepsize --net_1 $net1 --net_2 $net2
    """
}

AJI = file("ComputeAJI.py")
AnalyseFolder.concat(AnalyseFolderCaffe)
	     .set{ WaitingForAJImodels }

process ComputeAJI {
    clusterOptions = "-S /bin/bash"
    queue "all.q"
    publishDir "./Results", overwrite: true
    input:
    file py from AJI
    file fold from WaitingForAJImodels
    output:
    file fold into AnalysedAJI

    """
    python $py --fold $fold 
    """
}



process RegroupResults {
    clusterOptions = "-S /bin/bash"
    publishDir "./Results", overwrite: true

    input:
    file fold from AnalysedAJI .toList()
    output:
    file "results.csv" into RES

    """
    #!/usr/bin/env python

    from glob import glob
    import pandas as pd 
    from os.path import join
    from UsefulFunctions.RandomUtils import textparser

    folders = glob('Slide___*')
    result = pd.DataFrame(columns=["Image", "Model", "TP", "TN", "FN", "FP", "Precision", "Recall", "F1", "ACC", "AJI"])

    def name_parse(string):
        string = string.split('/')[-2]
        img_model = string.split('___')[1]
        return img_model.split('_*-_')

    for k, f in enumerate(folders):
        img, model = name_parse(join(f, "Characteristics.txt"))
        dic = textparser(join(f, "Characteristics.txt"))
        dic["Image"] = img
        dic["Model"] = model
        result.loc[k] = pd.Series(dic)

    result = result.set_index(["Model", "Image"]) 
    result.to_csv("results.csv")
    """
}

BARCHARTS = file("BarCharts.py")

process PlotBarCharts {
    clusterOptions = "-S /bin/bash"
    publishDir "./Results", overwrite: true
    input:
    file csv_file from RES
    file py from BARCHARTS
    output:
    file "BarPlotResult.png"
    

    """
    python $py
    """



}




