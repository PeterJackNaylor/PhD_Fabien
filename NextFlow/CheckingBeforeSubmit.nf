   
params.py = "PredictionSlide.py"
PYTHONFILE = file(params.py)

params.x = '0'
params.y = '0'
params.size_x = "224"
params.size_y = "224"
params.ref_level = '0'


PUBLISHDIR = "/share/data40T_v2/Peter/PatientFolder/"
params.slide = "579673.tiff"
SLIDE = file(params.slide)

params.size = "224"



process OneJob {
    executor 'sge'
	profile = 'cluster'
    validExitStatus 0, 134
    clusterOptions = "-S /bin/bash"
    publishDir PUBLISHDIR, overwrite: false


    input:
    val x from params.x
    val y from params.y
    val size_x from params.size_x
    val size_y from params.size_y
    val ref_level from params.ref_level
    val size from params.size
    file slide from SLIDE
    file pythonfile from PYTHONFILE

    output:
    file "Job_${slide.getBaseName()}/tiled/${x}_${y}_${size_x}_${size_y}_${ref_level}.tiff" into IMAGE_PROCESSED

    """
    python $pythonfile -x $x -y $y --size_x $size_x --size_y $size_y --ref_level $ref_level --output Job_${slide.getBaseName()}/ --slide $slide --size $size
    """


}



