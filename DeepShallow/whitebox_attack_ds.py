import os
import torch
import numpy as np

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs
import time
from profiler import profile_sdn, profile

from torch.autograd import Variable
import numpy as np
from torch import autograd
#torch.multiprocessing.set_start_method('spawn')
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from torch import optim
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import csv
import time




models_path='/glusterfs/data/mxh170530/Shallow-Deep-Networks-master/networks/1221/'
sdn_name='cifar10_mobilenet_sdn_sdn_training'
sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
sdn_model.to(device)

dataset = af.get_dataset('cifar10',1)
sdn_model.forward = sdn_model.early_exit
sdn_model.confidence_threshold = 0.9
sdn_model.eval()
def get_exit(outputs,bc):

    max_values=[]
    print(np.asarray(outputs).shape)
    time.sleep(10)
    for tmp in outputs:
        max_values.append(torch.max(tmp, 1))
    print(max_values)

    vals=[]
    indexes=[]
    for idx in range(bc):
        tmp=max_values[idx]
        for i in range(len(tmp.tolist())):
            if(tmp[0].tolist() > 0.9):
                vals.append(tmp[0])
                tmp.append(i+1)
                break
    print(indexes)
    print(vals)
    time.sleep(100000)
    return indexes,vals


def get_block_count(agent,inputs):
    probs, _ = agent(inputs)
    # print(inputs)
    policy = probs.clone()
    #print(policy)
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0
    sm=0
    off_blocks=[]
    lst = policy.tolist()[0]
    for ind in range(len(lst)):
        # print("l ",l)
        if (lst[ind] == 1):
            sm += 1
        else:
            off_blocks.append(ind)
    f = open("off_blocks.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    writer.writerow(off_blocks)
    f.close()

    return policy,sm


def tanh_rescale(x, x_min=-1.8, x_max=2.24):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    #print(x.size())
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)


def loss_op2(output, dist, scale_const):

    #sorted,indices=torch.sort(output,dim=1,descending=True)
    loss1 = output

    #print(loss1)
    #time.sleep(10)
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss

    #return loss1


def loss_op3(output, dist, scale_const):

    #sorted,indices=torch.sort(output,dim=1,descending=True)
    target=torch.tensor(2.30)
    loss1 = torch.clamp(target - output, min=0.)
    #loss1 = output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss


def loss_op4(output, dist, scale_const):

    #sorted,indices=torch.sort(output,dim=1,descending=True)
    #target=
    #target=torch.tensor(2.30)
    loss1 = output
    #loss1 = output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss
def loss_op(output, target, dist, scale_const):

    loss1 = torch.clamp(target - output, min=0.)
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss



#
hid_arr=[]
hid_arr2=[]
bc=100
hid_vals=[]
hid_vals_adv=[]
for batch in dataset.train_loader:
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    input_var = Variable(b_x,requires_grad=True).to(device)
    input_var.retain_grad()
    b_y = Variable(b_y,volatile=True).to(device)
    #preds, exit, is_early, inp = sdn_model(input_var)
    output = sdn_model(input_var)
    preds=output[0]
    inp=output[3]
    #print(output)
    #time.sleep(100000)
    hid_arr.append(input_var.tolist())
    tmp_inp=[inp[i].flatten().cpu().detach().numpy().tolist() for i in range(len(inp))]
    #print(tmp_inp)
    #time.sleep(100000)
    hid_vals.append(tmp_inp)
    orig=len(output)
    lst2 = np.zeros((orig, 32, 32), dtype=np.float64)
    ind = 0

    modifier = torch.zeros(input_var.size(), device=device).float()
    new_input_adv = torch.zeros(input_var.size()).float()
    #new_policy = torch.zeros(policy.size()).float()
    modifier_var = autograd.Variable(modifier, requires_grad=True)
    optimizer = optim.Adam([modifier_var], lr=0.0005)
    min_loss = float("inf")
    adv_img_min = np.zeros((1, 32, 32, 3))
    final_sm=0
    for ind in range(1000):
        x = torch.tensor(100000)
        scale_const_var = autograd.Variable(x, requires_grad=False).to(device)
        input_adv = tanh_rescale(modifier_var + input_var, -1.7, 2.05)
        output = sdn_model(input_adv)
        preds=output[0]

        softmax = torch.nn.functional.softmax(preds[0], dim=0)
        max_val=torch.max(softmax)
        #print(max_val)
        #time.sleep(100000)
        dist = l2_dist(input_adv, input_var, keepdim=False)
        loss = loss_op2(max_val, dist, scale_const_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        input_adv_np = input_adv.data.cpu().numpy()
        output = sdn_model(input_adv)
        preds = output[0]
        exit=output[1]
        inp = output[3]
        if(final_sm<exit):

            final_sm=exit
            print(final_sm)
            if(final_sm>=4):
                hid_arr2.append(input_adv_np)
                tmp_inp = [inp[i].flatten().cpu().detach().numpy().tolist() for i in range(len(inp))]
                hid_vals_adv.append(tmp_inp)
            if(final_sm==5):
                break
    np.save('ILFO/orig_C10_data_ILFO_mobilenet.npy',
                np.asarray(hid_arr))

    np.save('ILFO/adv_C10_data_ILFO_mobilenet.npy',
                np.asarray(hid_arr2))
    np.save('ILFO/orig_hid_C10_ILFO_mobilenet.npy',
            np.asarray(hid_vals))

    np.save('ILFO/adv_hid_C10_ILFO_mobilenet.npy',
            np.asarray(hid_vals_adv))
