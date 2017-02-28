#!/usr/bin/env nextflow

params.test = "Eureka"

VALUE = params.test

params.value = "Not Eureka"

VALUE = params.value

WD_REMOTE = "/share/data40T_v2/Peter"

params.text = ""
TEXT = file(params.text)

Channel
     .fromPath('eureka.txt')
     .splitText( by: 1 )
     .subscribe {
         print it;
         print "--- end of the chunk ---\n"
     }


process SecondNF {
    executor "local"
    queueSize = 30
	profile = 'cluster'
    validExitStatus 0,134
    clusterOptions = "-S /bin/bash"
    publishDir WD_REMOTE, mode: "copy", overwrite: false


    input:
    file "chunk" from TEXT.splitText()
    
    """
    nXame=`cat chunk`
    echo  \$nXame
    """


}