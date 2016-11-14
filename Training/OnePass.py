from options import GetOptions
import caffe

create_dataset = True
create_net = True
create_solver = True
train = True


if __name__ == "__main__":

    (options, args) = GetOptions()
    if create_dataset:
        from Data.DataGen import MakeDataGen
        MakeDataGen(options)

    if create_net:
        if options.net == "UNet":
            from Nets.UNet import make_net
            make_net(options)
        elif options.net == "DeconvNet":
            from Nets.DeconvNet import make_net
            make_net(options)
        elif options.net == "PangNet":
            from Nets.PangNet import make_net
            make_net(options)
        elif options.net == "FCN":
            for num in options.archi:
                if num == 8:
                    from Nets.FCN8 import make_net
                    make_net(options)
                elif num == 16:
                    from Nets.FCN16 import make_net
                    make_net(options)
                elif num == 32:
                    from Nets.FCN32 import make_net
                    make_net(options)

    if create_solver:
        from Solver import WriteSolver
        WriteSolver(options)

    if train:
        from TrainModel import TrainModel
        TrainModel(options)
