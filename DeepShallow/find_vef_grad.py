import os
import torch
import numpy as np

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs
import time
from profiler import profile_sdn, profile
os.environ["CUDA_VISIBLE_DEVICES"] = '7'


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

dataset = af.get_dataset('cifar100',1)
#sdn_model.forward = sdn_model.early_exit
#sdn_model.confidence_threshold = 0.9
sdn_model.eval()


i=0
for batch in dataset.train_loader:
    print(i)
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    #b_x = torch.autograd.Variable(b_x, requires_grad=True)
    input_var = torch.autograd.Variable(b_x, requires_grad=True).to(device)
    input_var.retain_grad()
    output,inp = sdn_model(input_var)
    lst=[]
    for o in output:
        #print(o)
        softmax=torch.nn.functional.softmax(o, dim=1)

        max_val=torch.max(softmax)
        if(max_val.tolist()>=0.9):
            lst.append(max_val)
    if(len(lst)==0):
        i+=1
        continue
    loss=get_loss(lst)
    inp.retain_grad()
    loss.backward()
    tmp1 = inp.grad.data.abs()
    if (i == 0):
        hid_arr = tmp1
    else:
        hid_arr = hid_arr + tmp1

    if (i % 1000 == 0):
        np.save('hid_c100_defad_org_grad.npy', np.asarray(hid_arr.tolist()))
    i += 1
np.save('hid_c100_defad_org_grad.npy', np.asarray(hid_arr.tolist()))