from solver import solver, run_solvers, run_solvers_IU
import os


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def WriteSolver(kwargs):
    """
    Compulsory arguments:
        wd : work directory
        cn : classifier name
        solverrate: solverrate

    Optionnal:
        enlarge : if to enlarge image when deforming
        loss : loss wished, specifically for the weight generations
        crop : if to crop
        crop_size : if random cropping
        img_format : can be RGB, HE, HEDab
        seed: seed to pick from


    """

    wd = kwargs['wd']
    cn = kwargs['cn']
    solverrate = kwargs['solverrate']

    print 'wd        ----   {}   ------'.format(str(wd))
    print 'cn        ----   {}   ------'.format(str(cn))
    print 'solverrate----   {}   ------'.format(str(solverrate))

    if "archi" in kwargs.keys():
        solver_path = os.path.join(options.wd, options.cn, "solver.prototxt")
        outsnap = os.path.join(options.wd, options.cn, "snapshot")

        CheckOrCreate(os.path.join(options.wd, options.cn))
        CheckOrCreate(outsnap)

        name_solver = solver(solver_path,
                             os.path.join(options.wd, options.cn,
                                          "train.prototxt"),
                             test_net_path=os.path.join(
                                 options.wd, options.cn, "train.prototxt"),
                             base_lr=solverrate,
                             out_snap=outsnap)
    else:
        for num in kwargs["archi"]:
            fcn_num = "FCN{}".format(num)
            solver_path = os.path.join(
                options.wd, options.cn, fcn_num, "solver.prototxt")
            outsnap = os.path.join(options.wd, options.cn, fcn_num, "snapshot")

            CheckOrCreate(os.path.join(options.wd, options.cn, fcn_num))
            CheckOrCreate(outsnap)

            name_solver = solver(solver_path,
                                 os.path.join(options.wd, options.cn,
                                              "train.prototxt"),
                                 test_net_path=os.path.join(
                                     options.wd, options.cn, "train.prototxt"),
                                 base_lr=solverrate,
                                 out_snap=outsnap)
