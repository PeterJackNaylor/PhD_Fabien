import subprocess
from UsefulFunctions.EmailSys import ElaborateEmail
from os import environ


net = 'FCN'
raw_data = "/data/users/pnaylor/Bureau/ToAnnotateColor"
wd = "/data/users/pnaylor/Documents/Python/Experiences2/MultiClass"
weight = "/data/users/pnaylor/Documents/Python/FCN/model/fcn32s-heavy-pascal.caffemodel"
niter = 200
disp_interval = 100
leaveout = 1
crop = 4

base_lr_list = [0.0001]  # , 0.001, 0.0001]

batch_size = 1
img_format = "RGB"
loss = 'softmax'

momentum_list = [0.9] #, 0.99]

weight_decay_list = [0.0005] #, 0.0005]

stepsize = 7000
gamma = 0.1
size_x = 224
size_y = 224
archi = "32_16_8"  # _16_8"

hw = "gpu"
num_output = 9
crf = 0


for base_lr in base_lr_list:
    for momentum in momentum_list:
        for weight_decay in weight_decay_list:
            cn = (net + '_{}_{}_{}').format(base_lr, momentum, weight_decay)
            arguments = (net, raw_data, wd, cn, weight, niter, disp_interval, leaveout, crop, base_lr,
                         batch_size, img_format, loss, momentum, weight_decay, stepsize, gamma, size_x, size_y, archi, hw, num_output, crf)
            cmd = "python Training/OnePass.py --net {} --rawdata {} --wd {} --cn {} --weight {} --niter {} --disp_interval {} --leaveout {} --crop {} --base_lr {} --batch_size {} --img_format {} --loss {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --size_x {} --size_y {} --archi {} --hw {} --num_output {} --crf {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()


body = "The job on {} using node {} is now free".format(environ["HOSTNAME"],environ["CUDA_VISIBLE_DEVICES"])
subject = "Free node"
ElaborateEmail(body, subject)
