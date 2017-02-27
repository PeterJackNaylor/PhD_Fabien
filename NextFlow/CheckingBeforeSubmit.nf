   
params.py = "/share/data40T_v2/Peter/PatientFolder/*/PredictionSlide.py"
PYTHONFILE = file(params.py)

params.x = '0'
params.y = '0'
params.size_x = "224"
params.size_y = "224"
params.ref_level = '0'

params.output = "/share/data40T_v2/Peter/PatientFolder/*/"
OUTPUT = file(params.output)

params.slide = "579673.tiff"
SLIDE = file(params.slide)

params.size = "224"



process OneJob {

	profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, overwrite: false


    input:
    val x from params.x
    val y from params.y
    val size_x from params.size_x
    val size_y from params.size_y
    val ref_level from params.ref_level
    file output from OUTPUT
    file slide from SLIDE
    file pythonfile from PYTHONFILE

    output:
    file $output/tiled/$x_$y_$size_x_$size_y_$ref_level.tiff

    """
    python $pythonfile -x $x -y $y --size_x $size_x --size_y $size_y --ref_level $ref_level --output $output --slide $slide --size $size -profiles cluster -resume
    """


}



