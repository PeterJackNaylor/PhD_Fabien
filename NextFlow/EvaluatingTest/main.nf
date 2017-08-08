#!/usr/bin/env nextflow

params.image_dir = '/data/users/pnaylor/Bureau'
params.python_dir = '/data/users/pnaylor/Documents/Python/PhD_Fabien'
params.home = "/data/users/pnaylor"

IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")
TENSORBOARD = file(params.image_dir + '/UNet3')
MEANPY = file(params.python_dir + '/NewStuff/MeanCalculation.py')

EXP = params.home + '/TF_EXP'

DataNeeraj = file(params.image_dir + "/NeerajKumar/ForDatagen")

ORGANS = ["Breast", "Bladder"]


process Mean {
    clusterOptions = "-S /bin/bash"

    input:
    file py from MEANPY
    file toannotate from IMAGE_FOLD
    output:
    file "mean_file.npy" into MeanFile

    """
    python $py --path $toannotate --output .
    """
}


process PrepareImages {
    clusterOptions = "-S /bin/bash"
    input:
    file input from DataNeeraj
    val organs from ORGANS

    output:
    file "Slide___*.png" into SLIDE_UNET mode flatten
    file "GT___*.png" into GT_UNET mode flatten
    """
    #!/usr/bin/env python

    from UsefulFunctions.ImageTransf import ListTransform
    from NewStuff.DataGenClass import DataGenMulti
    from scipy.misc import imsave
    _, transform_list_test = ListTransform()
    DG = DataGenMulti("$input", split='test', crop = 1, size=(1000, 1000),
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
 
UNET2_EXP = [EXP + "/Classic/32_0.00005_0.0001", EXP + "/NO_EL_HSV_HE/32_0.00005_0.0001"]
PARAM = 10
UNETPREDICTION = file('UNetPrediction.py')

process UNetBN_2 {
    clusterOptions = "-S /bin/bash"
    input:
    file py from UNETPREDICTION .last()
    file image from SLIDE_UNET
    file anno from GT_UNET
    file mean_file from MeanFile .first()
    val param from PARAM
    each fold from UNET2_EXP
    output:
    file "${image.getBaseName()}_*-_" into AnalyseFolder

    """
    PS1=\${PS1:=} CONDA_PATH_BACKUP="" source activate cpu_tf
    /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH /cbio/donnees/pnaylor/anaconda2/bin/python $py -i $image -a $anno -f $fold --mean_file $mean_file --param $param 
    """


}


AJI = file("ComputeAJI.py")

process ComputeAJI {
    clusterOptions = "-S /bin/bash"
    input:
    file py from AJI
    file fold from AnalyseFolder
    output:
    file fold into AnalyseFolder2

    """
    python $py --fold $fold 
    """

   




}
