import subprocess
from UsefulFunctions.EmailSys import ElaborateEmail
from os import environ

net = 'UNet'
raw_data = "/data/users/pnaylor/Bureau/Isbi2012/train-volume.tif"
wd = "/data/users/pnaylor/Documents/Python/Experiences2/Isbi2012LongWeights"
# weight = "/data/users/pnaylor/Documents/Python/FCN/model/DeconvNet_trainval_inference.caffemodel"
niter = 100000
disp_interval = 100
leaveout = 5


base_lr_list = [10 ** (-7)]
# base_lr_list = [10 **(i) for i in range(-6,0)]
batch_size = 1
loss = 'weightcpp'
momentum_list = [0.99]

weight_decay_list = [5 * 10 **(-5)]
# weight_decay_list = [5 * 10 **(-el)  for el in range(4,6)]

stepsize = 7000
gamma = 0.1

hw = "gpu"
mode = "Isbi2012"
w_0 = 10
val_b = 3
val_n = 1
sig_WGT = 5

for base_lr in base_lr_list:
    for momentum in momentum_list:
        for weight_decay in weight_decay_list:
            cn = (net + '_{}_{}_{}').format(base_lr, momentum, weight_decay)
            arguments = (net, raw_data, wd, cn, niter, disp_interval, leaveout, base_lr,
                         batch_size, loss, momentum, weight_decay, stepsize, gamma, hw, mode, w_0, val_b, val_n, sig_WGT)
            cmd = "python Training/OnePass.py --net {} --rawdata {} --wd {} --cn {} --niter {} --disp_interval {} --leaveout {} --base_lr {} --batch_size {} --loss {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --hw {} --mode {} --w_0 {} --val_b {} --val_n {} --sig_WGT {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()


body = "The job on {} using node {} is now free".format(environ["HOSTNAME"],environ["CUDA_VISIBLE_DEVICES"])
subject = "Free node"
ElaborateEmail(body, subject)
