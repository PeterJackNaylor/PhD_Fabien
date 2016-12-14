import subprocess
from UsefulFunctions.EmailSys import ElaborateEmail
from os import environ

net = 'UNet'
raw_data = "/data/users/pnaylor/Bureau/Isbi2012/train-volume.tif"
wd = "/data/users/pnaylor/Documents/Python/Experiences2"
# weight = "/data/users/pnaylor/Documents/Python/FCN/model/DeconvNet_trainval_inference.caffemodel"
niter = 3000
disp_interval = 1000
leaveout = 5


base_lr_list = [10**(-el) for el in range(1, 6)]
batch_size = 1

momentum_list = [0.9, 0.99]

weight_decay_list = [5 * 10 **(-el)  for el in range(3,7)]

stepsize = 7000
gamma = 0.1

hw = "gpu"
mode = "Isbi2012"
for base_lr in base_lr_list:
    for momentum in momentum_list:
        for weight_decay in weight_decay_list:
            cn = (net + '_{}_{}_{}').format(base_lr, momentum, weight_decay)
            arguments = (net, raw_data, wd, cn, niter, disp_interval, leaveout, base_lr,
                         batch_size, momentum, weight_decay, stepsize, gamma, hw, mode)
            cmd = "python Training/OnePass.py --net {} --rawdata {} --wd {} --cn {} --niter {} --disp_interval {} --leaveout {} --base_lr {} --batch_size {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --hw {} --mode {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()

body = "The job on {} using node {} is now free".format(environ["HOSTNAME"],environ["CUDA_VISIBLE_DEVICES"])
subject = "Free node"
ElaborateEmail(body, subject)