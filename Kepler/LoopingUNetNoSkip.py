import subprocess
from UsefulFunctions.EmailSys import ElaborateEmail
from os import environ

net = 'UNet'
raw_data = "/data/users/pnaylor/Bureau/ToAnnotate"
wd = "/data/users/pnaylor/Documents/Python/Experiences2/SmallIter"
# weight = "/data/users/pnaylor/Documents/Python/FCN/model/DeconvNet_trainval_inference.caffemodel"
niter = 3000
disp_interval = 100
leaveout = 1
crop = 4


base_lr_list = [10**(-el) for el in range(1, 6)]
batch_size = 1
img_format = "RGB"
loss = 'weightcpp'

momentum_list = [0.9, 0.99]

weight_decay_list = [5 * 10 **(-el)  for el in range(3,7)]

stepsize = 7000
gamma = 0.1
size_x = 212
size_y = 212

hw = "gpu"

w_0 = 10
val_b = 1
val_n = 3
sig_WGT = 10

for base_lr in base_lr_list:
    for momentum in momentum_list:
        for weight_decay in weight_decay_list:
            cn = (net + '_{}_{}_{}').format(base_lr, momentum, weight_decay)
            arguments = (net, raw_data, wd, cn, niter, disp_interval, leaveout, crop, base_lr,
                         batch_size, img_format, loss, momentum, weight_decay, stepsize, gamma, size_x, size_y, hw, w_0, val_b, val_n, sig_WGT)
            cmd = "python Training/OnePass.py --net {} --rawdata {} --wd {} --cn {} --niter {} --disp_interval {} --leaveout {} --crop {} --base_lr {} --batch_size {} --img_format {} --loss {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --size_x {} --size_y {} --hw {} --w_0 {} --val_b {} --val_n {} --sig_WGT {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()

body = "The job on {} using node {} is now free".format(environ["HOSTNAME"],environ["CUDA_VISIBLE_DEVICES"])
subject = "Free node"
ElaborateEmail(body, subject)
