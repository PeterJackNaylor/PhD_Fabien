
DATA = file('scatteringlengthsimoninew.dat')
GPELAB = file('GPELab')
MATLAB_FILE = file('GroundStateWithLHYTermSingleComponent_SolitonDroplet.m')
MATLAB_NAME = 'GroundStateWithLHYTermSingleComponent_SolitonDroplet'

COLLECT_MAT = file('Regroup.m')
COLLECT_MAT_NAME = 'Regroup'
BPointDiagram = 100
Spacing = 9

BEGINING = Channel.from( 1, 11, 21, 31, 41) 
//BEGINING = Channel.from( 1, 11, 21, 31, 41]) //\\, 51, 61, 71, 81, 91 )

ENDING   = Channel.from( 10,20, 30, 40, 50) 
//ENDING   = Channel.from( 10,20, 30, 40, 50]) //\\, 60, 70, 80, 90, 100 )
process Compute_J {
    memory = '10GB'
    cpus 10
    maxForks 16
    publishDir "results_1", overwrite: false
    input:
    file data from DATA
    file gpelab from GPELAB
    file matlab_file from MATLAB_FILE
    val matlab_name from MATLAB_NAME
    val bpointdiagram from BPointDiagram
    val beg from BEGINING
    val end from ENDING
    output:
    file "PhaseDiagram_nmax_*.mat" into NMAX
    file "PhaseDiagram_Nat.mat" into Nat
    file "PhaseDiagram_BVec.mat" into BVec

    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_name} $bpointdiagram $beg $end;exit;'
    """


}
BPointDiagram2 = 50
process Regroup {
    publishDir "final_results", overwrite: false
    input:
    file _ from NMAX .collect()
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
