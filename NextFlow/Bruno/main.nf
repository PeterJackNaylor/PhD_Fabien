
DATA = file('scatteringlengthsimoninew.dat')
GPELAB = file('GPELab')
MATLAB_FILE = file('GroundStateWithLHYTermSingleComponent_SolitonDroplet.m')
COLLECT_MAT = file('Regroup.m')
MATLAB_NAME = 'GroundStateWithLHYTermSingleComponent_SolitonDroplet'
SIZE = 32

process Compute_J {
    publishDir "results_j", overwrite: false
    input:
    file data from DATA
    file gpelab from GPELAB
    file matlab_file from MATLAB_FILE
    each col from 1..SIZE
    output:
    file "PhaseDiagram_nmax_*.mat" into NMAX
    file "PhaseDiagram_Nat_*.mat" into Nat
    file "PhaseDiagram_BVec_*.mat" into BVec

    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_file} $col;exit;'
    """

}

process Regroup {
    publishDir "final_results", overwrite: false
    input:
    file _ from NMAX .collect()
    file matlab_file from COLLECT_MAT
    output:
    file "FinalMat.mat"
    """
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_file.split('.')[0]};exit;'
    """


}