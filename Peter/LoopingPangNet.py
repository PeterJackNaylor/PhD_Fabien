import subprocess

net = 'PangNet'
raw_data = "/home/pnaylor/Documents/Data/ToAnnotate"
wd = "/home/pnaylor/Documents/Experiences/PangNet"
niter = 10000
disp_interval = 100
leaveout = 1
crop = 4


base_lr_list = [100, 10, 1, 0.01, 0.001]
batch_size = 20
img_format = "RGB"
loss = 'softmax'

momentum_list = [0.9, 0.99]

weight_decay_list = [0.0005, 0.00005]

stepsize = 7000
gamma = 0.1
size_x = 224
size_y = 224

hw = "gpu"

for base_lr in base_lr_list:
    for momentum in momentum_list:
        for weight_decay in weight_decay_list:
            cn = (net + '_{}_{}_{}').format(base_lr, momentum, weight_decay)
            arguments = (net, raw_data, wd, cn, niter, disp_interval, leaveout, crop, base_lr,
                         batch_size, img_format, loss, momentum, weight_decay, stepsize, gamma, size_x, size_y, hw)
            cmd = "python Training/OnePass.py --net {} --rawdata {} --wd {} --cn {} --niter {} --disp_interval {} --leaveout {} --crop {} --base_lr {} --batch_size {} --img_format {} --loss {} --momentum {} --weight_decay {} --stepsize {} --gamma {} --size_x {} --size_y {} --hw {}".format(
                *arguments)
            proces = subprocess.Popen(cmd, shell=True)
            proces.wait()
