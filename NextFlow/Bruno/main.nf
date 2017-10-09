
DATA = file('scatteringlengthsimoninew.dat')
GPELAB = file('GPELab')
MATLAB_FILE = file('GroundStateWithLHYTermSingleComponent_SolitonDroplet.m')
MATLAB_NAME = 'GroundStateWithLHYTermSingleComponent_SolitonDroplet'
MATLAB_FILE2 = file('GoingUpGround.m')
MATLAB_NAME2 = 'GoingUpGround'

COLLECT_MAT = file('Regroup.m')
COLLECT_MAT_NAME = 'Regroup'
BPointDiagram = 100
Spacing = 9

//BEGINING = Channel.from( 51, 61, 71, 81, 91) 
//BEGINING2 = Channel.from( 51, 61, 71, 81, 91) 
//BEGINING = Channel.from( 1, 11, 21, 31, 41) 
BEGINING = Channel.from( 1, 11, 21, 31, 41, 51, 61, 71, 81, 91 )
BEGINING2 = Channel.from( 1, 11, 21, 31, 41, 51, 61, 71, 81, 91 )
//ENDING   = Channel.from( 60, 70, 80, 90, 100) 
//ENDING2   = Channel.from( 60, 70, 80, 90, 100) 
//ENDING   = Channel.from( 10,20, 30, 40, 50) 
ENDING   = Channel.from( 10,20, 30, 40, 50, 60, 70, 80, 90, 100 )
ENDING2   = Channel.from( 10,20, 30, 40, 50, 60, 70, 80, 90, 100 )

process Compute_JDown {
    memory = '10GB'
    cpus 10
    maxForks 16
    publishDir "results_down", overwrite: true
    input:
    file data from DATA
    file gpelab from GPELAB
    file matlab_file from MATLAB_FILE
    val matlab_name from MATLAB_NAME
    val bpointdiagram from BPointDiagram
    val beg from BEGINING
    val end from ENDING
    output:
    file "PhaseDiagram_nmax_*.mat" into NMAXDown
    file "PhaseDiagram_Nat.mat" into NatDown
    file "PhaseDiagram_BVec.mat" into BVecDown

    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_name} $bpointdiagram $beg $end;exit;'
    """
}
process Compute_JUp {
    memory = '10GB'
    cpus 15
    maxForks 16
    publishDir "results_up", overwrite: true
    input:
    file data from DATA
    file gpelab from GPELAB
    file matlab_file from MATLAB_FILE2
    val matlab_name from MATLAB_NAME2
    val bpointdiagram from BPointDiagram
    val beg from BEGINING2
    val end from ENDING2
    output:
    file "PhaseDiagram_nmax_*.mat" into NMAXUp
    file "PhaseDiagram_Nat.mat" into NatUp
    file "PhaseDiagram_BVec.mat" into BVecUp

    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_name} $bpointdiagram $beg $end;exit;'
    """
}

BPointDiagram2 = 100
process RegroupDown {
    publishDir "results_down", overwrite: true
    input:
    file _ from NMAXDown .collect()
    file matlab_file from COLLECT_MAT
    val matlab_file_name from COLLECT_MAT_NAME
    val bpointdiagram from BPointDiagram2
    val space from Spacing
    output:
    file "FinalMat.mat"
    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_file_name} $bpointdiagram $space;exit;'
    """
}
process RegroupUp {
    publishDir "results_up", overwrite: true
    input:
    file _ from NMAXUp .collect()
    file matlab_file from COLLECT_MAT
    val matlab_file_name from COLLECT_MAT_NAME
    val bpointdiagram from BPointDiagram2
    val space from Spacing
    output:
    file "FinalMat.mat"
    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_file_name} $bpointdiagram $space;exit;'
    """
}
