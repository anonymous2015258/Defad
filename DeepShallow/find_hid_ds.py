import os
import torch
import numpy as np

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs
import time
from torch.autograd import Variable
import torch.utils.data as Data
from profiler import profile_sdn, profile
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

#mean = [0.4914, 0.4824, 0.4467]
#std = [0.2471, 0.2435, 0.2616]
#mean=[0.485, 0.456, 0.406]
#std=[0.229, 0.224, 0.225]

mean=[0.507, 0.487, 0.441]
std=[0.267, 0.256, 0.276]

def normalize(t):
    n = t.clone()
    for i in range(3):
        n[:, i, :, :] = (t[:, i, :, :] - mean[i]) / std[i]
    return n

def get_loss(lst):

    sm=torch.tensor(0.0)
    for i in range(len(lst)):
        if(i==0):
            sm=lst[i]-torch.tensor(0.9)
        else:
            tmp=lst[i]-torch.tensor(0.9)
            sm=torch.add(sm,tmp)
    return sm

device=torch.device('cuda')
models_path='/glusterfs/data/mxh170530/Shallow-Deep-Networks-master/networks/1221/'
#sdn_name='cifar10_vgg16bn_sdn_sdn_training'
sdn_name='cifar100_vgg16bn_sdn_sdn_training'
sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
sdn_model.to(device)

data=torch.load('ShallowDeep_vgg16_CIFAR100_linf.tar')
print(data.keys())

x=data['train_adv']
y=data['train_y']
x=torch.tensor(x)
size=x.size()
x=x.reshape((size[0]*size[1],3,32,32))

print(x.size())
x=normalize(x)
#data[0].size()
y=torch.tensor([0 for i in range(size[0]*size[1])])
x, y = Variable(x), Variable(torch.tensor(y))

torch_dataset = Data.TensorDataset(x, y)

adv_train_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=1,
    shuffle=False, num_workers=2, )

x=data['test_adv']
y=data['test_y']
x=torch.tensor(x)
size=x.size()
x=x.reshape((size[0]*size[1],3,32,32))

print(x.size())
x=normalize(x)
#data[0].size()
y=torch.tensor([0 for i in range(size[0]*size[1])])

x, y = Variable(x), Variable(y)

torch_dataset = Data.TensorDataset(x, y)

adv_test_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=1,
    shuffle=False, num_workers=2, )

#time.sleep(100000)
dataset = af.get_dataset('cifar100',1)
#sdn_model.forward = sdn_model.early_exit
#sdn_model.confidence_threshold = 0.9
sdn_model.eval()


i=0
hid_vals=[]
for batch in adv_train_loader:
    print(i)
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    #b_x = torch.autograd.Variable(b_x, requires_grad=True)
    input_var = torch.autograd.Variable(b_x, requires_grad=True).to(device)
    input_var.retain_grad()
    output,inp = sdn_model(input_var)
    tmp_inp = [inp[i].flatten().cpu().detach().numpy().tolist() for i in range(len(inp))]
    # print(tmp_inp)
    # time.sleep(100000)
    hid_vals.append(tmp_inp)

    if (i % 1000 == 0):
        np.save('hid_vals_deepsloth_c100.npy', np.asarray(hid_vals))
    i += 1
np.save('hid_vals_deepsloth_c100.npy', np.asarray(hid_vals))


hid_vals=[]
for batch in adv_test_loader:
    print(i)
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    #b_x = torch.autograd.Variable(b_x, requires_grad=True)
    input_var = torch.autograd.Variable(b_x, requires_grad=True).to(device)
    input_var.retain_grad()
    output,inp = sdn_model(input_var)
    tmp_inp = [inp[i].flatten().cpu().detach().numpy().tolist() for i in range(len(inp))]
    # print(tmp_inp)
    # time.sleep(100000)
    hid_vals.append(tmp_inp)

    if (i % 1000 == 0):
        np.save('hid_vals_deepsloth_c100_test.npy', np.asarray(hid_vals))
    i += 1
np.save('hid_vals_deepsloth_c100_test.npy', np.asarray(hid_vals))