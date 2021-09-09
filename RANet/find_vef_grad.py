import os
import math
import time
import shutil
import models
import time
from dataloader import get_dataloaders
from args import args
#from adaptive_inference import dynamic_evaluate
#from op_counter import measure_model
#from tx2_predict import PowerLogger,getNodes
#import csv
from torch.autograd import Variable
import torch.utils.data as Data
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pickle
#torch.manual_seed(args.seed)
import numpy as np
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


'''sal1=np.load('ILFO/orig_C10_data_ILFO.npy',allow_pickle=True)
#sal1=np.load('ILFO/adv_C10_data_ILFO.npy',allow_pickle=True)
sal1=sal1.reshape((len(sal1),3,32,32))
target_arr=[0 for i in range(len(sal1))]
x, y = Variable(torch.Tensor(sal1)), Variable(torch.Tensor(target_arr))

torch_dataset = Data.TensorDataset(x, y)

orig_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=1,
    shuffle=False, num_workers=2, )

sal1=np.load('ILFO/adv_C10_data_ILFO.npy',allow_pickle=True)
sal1=sal1.reshape((len(sal1),3,32,32))
target_arr=[0 for i in range(len(sal1))]
x, y = Variable(torch.Tensor(sal1)), Variable(torch.Tensor(target_arr))

torch_dataset = Data.TensorDataset(x, y)

adv_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=1,
    shuffle=False, num_workers=2, )'''

def get_loss(lst):

    sm=torch.tensor(0.0)
    for i in range(len(lst)):
        if(i==0):
            sm=lst[i][0]-torch.tensor(0.9)
        else:
            tmp=lst[i][0]-torch.tensor(0.9)
            sm=torch.add(sm,tmp)
    return sm

train_loader, val_loader, test_loader = get_dataloaders(args)
model = getattr(models, 'RANet')(args)
model = torch.nn.DataParallel(model.cuda())
criterion = nn.CrossEntropyLoss().cuda()

#train_loader, val_loader, test_loader = get_dataloaders(args)
#state_dict = torch.load('model_best2.pth.tar')
state_dict = torch.load('model_best_c100_2.pth.tar')
model.load_state_dict(state_dict)
model.eval().cuda()
def validate(train_loader, model):
    #batch_time = AverageMeter()
    #losses = AverageMeter()
    #data_time = AverageMeter()
    top1, top5 = [], []


    #model.eval()

    #end = time.time()
    #hid_arr=[]
    #hid_arr2=[]

    #with torch.no_grad():
    for i, (input, target) in enumerate(train_loader):
        model.eval().cuda()
        target = target.cuda(non_blocking=True)
        # input = input.cuda()
        # , requires_grad=True
        input_var = torch.autograd.Variable(input,requires_grad=True)
        target_var = torch.autograd.Variable(target)

        lst,inp = model(input)
        print(i)
        loss=get_loss(lst)
        inp[0].retain_grad()
        inp[1].retain_grad()
        #inp[2].retain_grad()
        #inp[3].retain_grad()
        loss.backward()
        tmp1=inp[0].grad.data.abs()
        tmp2=inp[1].grad.data.abs()
        #tmp3 = inp[1].grad.data.abs()
        #tmp4 = inp[1].grad.data.abs()
        if(i==0):
            hid_arr=tmp1
            hid_arr2=tmp2
        else:
            hid_arr=hid_arr+tmp1
            hid_arr2=hid_arr2+tmp2
        #print(output)
        #print(inp[0].size())
        #print(inp[1].size())
        #print(inp[2].size())
        #print(inp[3].size())
        #print(inp[4].size())
        '''if(i==2):
            time.sleep(10)
            break'''
        #print(inp[0].flatten().size())
        #hid_arr2.append(inp[0].flatten().tolist())
        #print(hid_arr)
        if(i%1000==0):
            np.save('hid_c100_defad_org_grad_l1.npy', np.asarray(hid_arr.tolist()))
            np.save('hid_c100_defad_org_grad_l2.npy', np.asarray(hid_arr2.tolist()))
    np.save('hid_c100_defad_org_grad_l1.npy',np.asarray(hid_arr.tolist()))
    np.save('hid_c100_defad_org_grad_l2.npy', np.asarray(hid_arr2.tolist()))



validate(train_loader,model)