#!/usr/bin/env nextflow
//params.in = files("/Users/naylorpeter/Documents/Histopathologie/NeerajKumar/TissueImages/*.tif")

params.in = files("/Users/naylorpeter/*.tif")

process Probability {
    maxForks = 1
    input:
    file img from params.in
    output:
    file "prob_*.tif" into PROB
    file "rgb_*.tif" into RGB, RGB2
    """
    cp $img prob_$img
    cp $img rgb_$img
    """
}

process Bin {
    maxForks = 1
    input:
    file prob from PROB
    file rgb from RGB2
    output:
    file "bin_*.tif" into BIN
    """
    #!/usr/bin/env python

    from shutil import copyfile

    copyfile('$prob', '$prob'.replace('prob', 'bin'))

    """
}
def getPositionID( file ) {
      file.name.split('_').drop(1).join('_')
}
RGB .phase(BIN, remainder: true) { it -> getPositionID(it) } .set { RGB_AND_BIN }


process Extract {
    input:
    set file(rgb), file(bin) from RGB_AND_BIN

    """
    #!/usr/bin/env python

    if '$bin'.replace('bin_prob', 'bin') != '$rgb'.replace('rgb', 'bin'):
        import WOUNA
    """
}