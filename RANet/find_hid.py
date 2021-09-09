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


sal1=np.load('ILFO/orig_C10_data_ILFO.npy',allow_pickle=True)
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
    shuffle=False, num_workers=2, )


model = getattr(models, 'RANet')(args)
model = torch.nn.DataParallel(model.cuda())
criterion = nn.CrossEntropyLoss().cuda()

#train_loader, val_loader, test_loader = get_dataloaders(args)
state_dict = torch.load('model_best2.pth.tar')
#state_dict = torch.load('model_best_c100_2.pth.tar')
model.load_state_dict(state_dict)
model.eval().cuda()
def validate(orig_loader,adv_loader, model):
    #batch_time = AverageMeter()
    #losses = AverageMeter()
    #data_time = AverageMeter()
    top1, top5 = [], []


    #model.eval()

    #end = time.time()
    hid_arr=[]
    hid_arr2=[]

    #with torch.no_grad():
    for i, (input, target) in enumerate(orig_loader):
        print(i)
        model.eval().cuda()
        target = target.cuda(non_blocking=True)
        # input = input.cuda()
        # , requires_grad=True
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output,inp = model(input)
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
        hid_arr.append(inp[0].flatten().tolist())
        hid_arr2.append(inp[1].flatten().tolist())
        print(inp[1].size())
        if(i%1000==0):
            #np.save('hid_c100_defad_org_l1.npy',np.asarray(hid_arr))
            np.save('hid_c10_defad_org_l2.npy', np.asarray(hid_arr2))
    #np.save('hid_c100_defad_org_l1.npy', np.asarray(hid_arr))
    np.save('hid_c10_defad_org_l2.npy', np.asarray(hid_arr2))
    hid_arr = []
    hid_arr2 = []
    for i, (input, target) in enumerate(adv_loader):
        print(i)
        model.eval().cuda()
        target = target.cuda(non_blocking=True)
        # input = input.cuda()
        # , requires_grad=True
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output,inp = model(input)
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
        hid_arr.append(inp[0].flatten().tolist())
        hid_arr2.append(inp[1].flatten().tolist())
        if (i % 1000 == 0):
            np.save('hid_c10_defad_adv_l1.npy', np.asarray(hid_arr))
            np.save('hid_c10_defad_adv_l2.npy', np.asarray(hid_arr2))
    np.save('hid_c100_defad_adv_l1.npy', np.asarray(hid_arr))
    np.save('hid_c10_defad_adv_l2.npy', np.asarray(hid_arr2))




validate(orig_loader,adv_loader,model)