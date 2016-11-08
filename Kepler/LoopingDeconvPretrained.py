import subprocess

net = 'UNet'
raw_data = "/data/users/pnaylor/Bureau/ToAnnotate"
wd = "/data/users/pnaylor/Documents/Python/Experiences2"
weight = "/data/users/pnaylor/Documents/Python/FCN/model/DeconvNet_trainval_inference.caffemodel"
niter = 300
disp_interval = 10
leaveout = 1
crop = 4


base_lr_list = [0.01]  # , 1, 0.01, 0.001, 0.0001]
batch_size = 4
img_format = "RGB"
loss = 'softmax'

momentum_list = [0.5, 0.7, 0.9]

weight_decay_list = [0.0005, 0.005]  # , 0.00005]

stepsize = 7000
gamma = 0.1
size_x = 224
size_y = 224

for base_lr in base_lr_list:
    for momentum in momentum_list:
        for weight_decay in weight_decay_list:
            cn = (net + '_{}_{}_{}').format(base_lr, momentum, weight_decay)
            arguments = (net, raw_data, wd, cn, weight, niter, disp_interval, leaveout, crop, base_lr,
                         batch_size, img_format, loss, momentum, weight_decay, stepsize, gamma, size_x, size_y)
            cmd = "python Training/OnePass.py --net {} --rawdata {} --wd {} --cn {} --weight {} --niter {} --disp_interval {} --leaveout {} --crop {} --base_lr {} --batch_size {} --img_format {} --loss {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --size_x {} --size_y {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()
