import subprocess
from UsefulFunctions.EmailSys import ElaborateEmail
from os import environ

net = 'UNet'
raw_data = "/home/pnaylor/Documents/Data/isbi2012/train-volume.tif"
wd = "/home/pnaylor/Documents/Experiences/temp_place/Isbi2012_long"
# weight = "/data/users/pnaylor/Documents/Python/FCN/model/DeconvNet_trainval_inference.caffemodel"
niter = 10
disp_interval = 1
leaveout = 5


base_lr_list = [0.01]
batch_size = 1

momentum_list = [0.9]

weight_decay_list = [0.00005] #5 * 10 **(-el)  for el in range(4,6)]

stepsize = 7000
gamma = 0.1
loss = "weightcpp"
hw = "cpu"
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
                         batch_size, momentum, weight_decay, stepsize, gamma, hw, mode, loss, w_0, val_b, val_n, sig_WGT)
            cmd = "python Training/OnePass.py --net {} --rawdata {} --wd {} --cn {} --niter {} --disp_interval {} --leaveout {} --base_lr {} --batch_size {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --hw {} --mode {} --loss {} --w_0 {} --val_b {} --val_n {} --sig_WGT {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()

body = "The job on {} using node {} is now free".format(environ["HOSTNAME"],environ["CUDA_VISIBLE_DEVICES"])
subject = "Free node"
ElaborateEmail(body, subject)

##### HELLO people
