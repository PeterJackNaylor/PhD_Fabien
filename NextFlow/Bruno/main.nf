
DATA = file('scatteringlengthsimoninew.dat')
GPELAB = file('GPELab')
MATLAB_FILE = file('GroundStateWithLHYTermSingleComponent_SolitonDroplet.m')
MATLAB_NAME = 'GroundStateWithLHYTermSingleComponent_SolitonDroplet'

COLLECT_MAT = file('Regroup.m')
COLLECT_MAT_NAME = 'Regroup'
SIZE = 32

process Compute_J {
    memory = '5GB'
    maxForks 16
    publishDir "results_j", overwrite: false
    input:
    file data from DATA
    file gpelab from GPELAB
    file matlab_file from MATLAB_FILE
    val matlab_name from MATLAB_NAME
    each col from 1..SIZE
    output:
    file "PhaseDiagram_nmax_*.mat" into NMAX
    file "PhaseDiagram_Nat_*.mat" into Nat
    file "PhaseDiagram_BVec_*.mat" into BVec

    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_name} $col;exit;'
    """

}

process Regroup {
    publishDir "final_results", overwrite: false
    input:
    file _ from NMAX .collect()
    file matlab_file from COLLECT_MAT
    val matlab_file_name from COLLECT_MAT_NAME
    file __ from Nat .collect()
    file ___ from BVec .collect()
    output:
    file "FinalMat.mat"
    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_file_name};exit;'
    """


}
