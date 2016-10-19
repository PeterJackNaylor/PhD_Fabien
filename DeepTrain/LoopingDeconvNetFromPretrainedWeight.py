import subprocess

net = 'DeconvNet'
raw_data = "/data/users/pnaylor/Bureau/ToAnnotate"
wd = "/data/users/pnaylor/Documents/Python/LoopingDeconvNetFromPretrainedWeight"
weight = "/data/users/pnaylor/Documents/Python/FCN/model/DeconvNet_trainval_inference.caffemodel"
niter = 5000
disp_interval = 100
epoch = 10
val_num = "1"
crop = "4"

solverrate_list = [1, 0.1, 0.01, 0.001, 0.0001]

batch_size = "4"
img_format = "RGB"
loss = 'softmax'

momentum_list = [0.9, 0.99]

weight_decay_list = [0.005, 0.0005, 0.00005]

stepsize = 10000
gamma = 0.1
size_x = 224
size_y = 224


for solverrate in solverrate_list:
    for momentum in momentum_list:
        for weight_decay in weight_decay_list:
            cn = (net + '_{}_{}_{}').format(solverrate, momentum, weight_decay)
            arguments = (net, raw_data, wd, cn, weight, niter, disp_interval, val_num, crop, solverrate,
                         batch_size, img_format, loss, momentum, weight_decay, stepsize, gamma, size_x, size_y, epoch)
            cmd = "python DeepTrain/runNet.py --net {} --rawdata {} --wd {} --cn {} --weight {} --niter {} --disp_interval {} --val_num {} --crop {} --solverrate {} --batch_size {} --img_format {} --loss {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --size_x {} --size_y {} --epoch {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()
