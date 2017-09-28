
DATA = file('scatteringlengthsimoninew.dat')
GPELAB = file('GPELab')
MATLAB_FILE = file('GroundStateWithLHYTermSingleComponent_SolitonDroplet.m')

SIZE = 10

process Compute_J {

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
    matlab -nodisplay -nosplash -nodesktop -r '${matlab_file.split('.')[0]} $col;exit;'
    """

}

