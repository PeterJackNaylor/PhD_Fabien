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
    momentum = kwargs['momentum']
    weight_decay = kwargs["weight_decay"]
    gamma = kwargs["gamma"]
    stepsize = kwargs["stepsize"]

    print 'wd          ----   {}   ------'.format(str(wd))
    print 'cn          ----   {}   ------'.format(str(cn))
    print 'solverrate  ----   {}   ------'.format(str(solverrate))
    print 'momentum    ----   {}   ------'.format(str(momentum))
    print 'weight_decay----   {}   ------'.format(str(weight_decay))
    print 'gamma       ----   {}   ------'.format(str(gamma))
    print 'stepsize    ----   {}   ------'.format(str(stepsize))

    if "archi" not in kwargs.keys():
        solver_path = os.path.join(wd, cn, "solver.prototxt")
        outsnap = os.path.join(wd, cn, "snapshot")

        CheckOrCreate(os.path.join(wd, cn))
        CheckOrCreate(outsnap)
        train_net_path = os.path.join(wd, cn, "train.prototxt")
        test_net_path = os.path.join(wd, cn, "test.prototxt")
        name_solver = solver(solver_path,
                             train_net_path,
                             test_net_path=test_net_path,
                             base_lr=solverrate,
                             out_snap=outsnap,
                             momentum=momentum,
                             weight_decay=weight_decay,
                             gamma=gamma,
                             stepsize=stepsize)
    elif len(kwargs['archi']) == 1:
        solver_path = os.path.join(wd, cn, "solver.prototxt")
        outsnap = os.path.join(wd, cn, "snapshot")
        num = kwargs['archi'][0]
        fcn_num = "FCN{}".format(num)

        CheckOrCreate(os.path.join(wd, cn))
        CheckOrCreate(outsnap)

        train_net_path = os.path.join(wd, cn, fcn_num, "train.prototxt")
        test_net_path = os.path.join(wd, cn, fcn_num, "test.prototxt")
        name_solver = solver(solver_path,
                             train_net_path,
                             test_net_path=test_net_path,
                             base_lr=solverrate,
                             out_snap=outsnap,
                             momentum=momentum,
                             weight_decay=weight_decay,
                             gamma=gamma,
                             stepsize=stepsize)

    else:
        for num in kwargs["archi"]:
            fcn_num = "FCN{}".format(num)
            solver_path = os.path.join(wd, cn, fcn_num, "solver.prototxt")
            outsnap = os.path.join(wd, cn, fcn_num, "snapshot")

            CheckOrCreate(os.path.join(wd, cn, fcn_num))

            train_net_path = os.path.join(wd, cn, fcn_num, "train.prototxt")
            test_net_path = os.path.join(wd, cn, fcn_num, "test.prototxt")
            CheckOrCreate(outsnap)

            name_solver = solver(solver_path,
                                 train_net_path,
                                 test_net_path=test_net_path, ,
                                 base_lr=solverrate,
                                 out_snap=outsnap,
                                 momentum=momentum,
                                 weight_decay=weight_decay,
                                 gamma=gamma,
                                 stepsize=stepsize)
