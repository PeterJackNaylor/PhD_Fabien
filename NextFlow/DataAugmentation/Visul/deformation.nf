
params.epoch = 1 
params.image_dir = '/share/data40T_v2/Peter/Data'
IMAGE_FOLD = file(params.image_dir + "/ToAnnotate")

HE = [0.01, 0.07, 0.1, 0.15, 0.20, 0.25]
HSV = [0.01, 0.07, 0.1, 0.15, 0.20, 0.25]
ELAST1 = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
ELAST2 = [0.01, 0.02, 0.03, 0.04, 0.046875, 0.05, 0.06, 0.07]
ELAST3 = [0.01, 0.07, 0.1, 0.15, 0.20, 0.25]

PLOT = file('src/plot_deformation.py')

process CreateTFRecords_he {
    clusterOptions = "-S /bin/bash"
    publishDir 'DeformationVisual'
    input:
    file py from PLOT
    val epoch from params.epoch
    file path from IMAGE_FOLD
    each he1 from HE
    each he2 from HE
    output:
    file "HE_${he1}_${he2}" into HE_FOLDER
    """

    python $py --output HE_${he1}_${he2} --he1 $he1 --he2 $he2 --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --train
    """
}

process CreateTFRecords_hsv {
    clusterOptions = "-S /bin/bash"
    publishDir 'DeformationVisual'
    input:
    file py from PLOT
    val epoch from params.epoch
    file path from IMAGE_FOLD
    each hsv1 from HSV
    each hsv2 from HSV
    output:
    file "HSV_${hsv1}_${hsv2}" into HSV_FOLDER
    """

    python $py --output HSV_${hsv1}_${hsv2} --hsv1 $hsv1 --hsv2 $hsv2 --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --train
    """
}

process CreateTFRecords_elast {
    clusterOptions = "-S /bin/bash"
    publishDir 'DeformationVisual'
    input:
    file py from PLOT
    val epoch from params.epoch
    file path from IMAGE_FOLD
    each elast1 from ELAST1
    each elast2 from ELAST2
    each elast3 from ELAST3

    output:
    file "Elast_${elast1}_${elast2}_${elast3}" into ELAST_FOLDER
    """

    python $py --output Elast_${elast1}_${elast2}_${elast3} --elast1 $elast1 --elast2 $elast2 --elast3 $elast3 --path $path --crop 4 --UNet --size 212 --seed 42 --epoch $epoch --type Normal --train
    """
}