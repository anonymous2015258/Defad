import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import csv
import torch.utils.data as torchdata2
import numpy as np
from torch.autograd import Variable
from torch import optim
import time
from torch import autograd
import sys
import torch.nn.functional as F
import torch.utils.data as Data
from models import base
from training_model import FlatResNet32
import utils


#torch.multiprocessing.set_start_method('spawn')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')




transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

trainset = torchdata.CIFAR100(root='data/', train=True, download=True, transform=transform_train)
testset = torchdata.CIFAR100(root='data/', train=False, download=True, transform=transform_test)
train_loader = torchdata2.DataLoader(trainset, batch_size=1,shuffle=False)
test_loader = torchdata2.DataLoader(testset, batch_size=1,shuffle=False)
train_dataset_array = next(iter(train_loader))[0].numpy()
test_dataset_array = next(iter(test_loader))[0].numpy()

lst=[]
with open('cifar_training_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[2]))

target_arr = np.asarray(lst)
# target_arr=np.reshape(target_arr,(5000,1))
#target_arr2 = np.asarray(lst)[5000:]
# target_arr2=np.reshape(target_arr2,(5000,1))
# print(target_arr)

lst=[]

def get_block_count(agent,inputs):
    probs, v ,x1= agent(inputs)
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
    '''f = open("off_blocks.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    writer.writerow(off_blocks)
    f.close()'''

    return policy,sm,x1

with open('Cifar_Energy.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[2]))

target_arr2 = np.asarray(lst)
def tanh_rescale(x, x_min=0, x_max=1):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)


def loss_op2(output, dist, scale_const):

    sorted,indices=torch.sort(output,dim=1,descending=True)
    loss1 = sorted[0][0]-sorted[0][1]
    loss1 = torch.sum(scale_const * loss1)
    #loss2 = dist.sum()
    #loss = loss1 + loss2
    #return loss
    return loss1

def loss_op(output, target, dist, scale_const):

    loss1 = torch.clamp(target - output, min=0.)

    #print(loss1)
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss

'''layer_config = [18, 18, 18]
net = FlatResNet32(base.BasicBlock, layer_config, num_classes=1)

model = net
model.load_state_dict(torch.load('cifar_train2.pth',map_location='cpu'))
model.eval().to(device)'''

x, y = Variable(torch.Tensor(test_dataset_array)), Variable(torch.Tensor(target_arr2))

rnet, agent = utils.get_model('R32_C100')

#utils.load_checkpoint(rnet, agent, 'cv/cv/finetuned/R110_C10_gamma_10/ckpt_E_2000_A_0.936_R_1.95E-01_S_16.93_#_469.t7')
utils.load_checkpoint(rnet, agent, 'cv/cv/finetuned/R32_C100_gamma_5/ckpt_E_2000_A_0.687_R_-1.40E+00_S_13.06_#_131.t7')
rnet.eval().to(device)
agent.eval().to(device)

'''torch_dataset = Data.TensorDataset(x, y)
BATCH_SIZE = 1
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, )'''
orig=[]
adv=[]

hid_arr=[]
hid_arr2=[]

hid_vals=[]
hid_vals_adv=[]
for step, (batch_x, batch_y) in enumerate(test_loader):  # for each training step
    #print("epoch ", epoch)
    '''if(step==0):
        continue
        print('in')'''
        #time.sleep(100)
    '''if(step==500):
        break'''
    #input_var = Variable(batch_x,requires_grad=True,volatile=True).to(device)
    #batch_x.requires_grad_()


    input_var = Variable(batch_x, requires_grad=True).to(device)
    '''np.save('img_orig_' + str(step) + '_data.npy',
            input_var.cpu().detach().numpy())'''
    input_var.retain_grad()
    #input_var=input_var.retain_grad()
    b_y = Variable(batch_y,volatile=True).to(device)
    #probs, _ = agent(input_var)
    #policy, sm = get_block_count(agent, input_var)

    #print('sm ', sm)

    policy, sm,x1 = get_block_count(agent, input_var)

    hid_arr.append(input_var.tolist())
    hid_vals.append(x1.tolist())

    print('sm ',sm)
    orig.append(sm)
    #print(input_var.grad)

    '''preds = rnet.forward_single(input_var, policy.data.squeeze(0))
    print("preds")
    print(preds)
    #time.sleep(2)
    f = open("orig_preds_uni.csv", "a")
    writer = csv.writer(f)
    softm = torch.softmax(preds,dim = 1)
    # print(energy)
    preds = preds.data.cpu().numpy()
    # blocks.append(sm2)
    writer.writerow(preds)
    f.close()'''
    #policy = probs.clone()
    modifier = torch.zeros(input_var.size(), device=device).float()
    new_input_adv = torch.zeros(input_var.size()).float()
    new_policy = torch.zeros(policy.size()).float()
    modifier_var = autograd.Variable(modifier, requires_grad=True)
    optimizer = optim.Adam([modifier_var], lr=0.0005)
    # target = torch.tensor(0.0)
    # target_var = autograd.Variable(target, requires_grad=False)
    min_loss = float("inf")
    adv_img_min = np.zeros((1, 32, 32, 3))
    best_saliency = np.zeros((32, 32))
    min_output = torch.zeros(policy.size()).float()
    #target = torch.tensor([1.0] * 54).to(device)
    target = torch.tensor([1.0] * 15).to(device)
    final_sm=0
    for ind in range(250):
        print(ind)
        x = torch.tensor(10000)
        scale_const_var = autograd.Variable(x, requires_grad=False).to(device)
        input_adv = tanh_rescale(modifier_var + input_var, 0, 1)
        policy, adv_sm,x1 = get_block_count(agent, input_adv)
        #preds = rnet.forward_single(input_adv, policy.data.squeeze(0))
        #probs, _ = agent(input_adv)
        # print(inputs)
        #policy = probs.clone()
        # print(output)

        dist = l2_dist(input_adv, input_var, keepdim=False)
        probs, v,x1= agent(input_adv)
        loss = loss_op(probs,target, dist, scale_const_var)
        #loss = loss_op2(preds, dist, scale_const_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        output_np = policy.data.cpu().numpy()
        # print(input_adv.data)
        input_adv_np = input_adv.data.cpu().numpy()
        # print(input_adv_np)

        policy, sm2 ,x1= get_block_count(agent, input_adv)
        print("sm2 ", sm2, " ", final_sm)
        if (final_sm < sm2):
            final_sm = sm2
            min_loss = loss
            adv_img_min = input_adv_np
            if(sm2>=13):
                hid_arr2.append(adv_img_min.tolist())
                adv.append(final_sm)
                hid_vals_adv.append(x1.tolist())
            '''if(sm2==25):
                break'''
            new_input_adv = input_adv


            #policy, sm = get_block_count(agent, input_var)

            new_policy=policy
            print("preds")
            #time.sleep(2)

            #print(preds)
            #if(sm2>final_sm):


            #min_output = policy
            print("min ", ind, " ", sm2)

            '''f = open("block2.csv", "a")
            writer = csv.writer(f)
            # print(energy)
            blocks = []
            blocks.append(sm2)
            writer.writerow(blocks)
            f.close()'''
            # time.sleep(2)
    # print("min ", min_output)
    #preds = rnet.forward_single(new_input_adv, new_policy.data.squeeze(0))
    '''print('time')
    start=time.time()
    probs, _ = agent(input_var)
    end = time.time()'''
    '''print(end-start)
    start = time.time()
    preds = rnet.forward_single(new_input_adv, new_policy.data.squeeze(0))
    end = time.time()
    print(end - start)'''
    # print(inputs)
    # policy = probs.clone()
    # print(policy[0])

    '''policy = probs.clone()
    # print(policy)
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0
    lst = policy.tolist()[0]'''



    '''adv_clone=new_input_adv.clone().detach()

    adv_clone = autograd.Variable(adv_clone, requires_grad=True).to(device)
    adv_clone.retain_grad()
    probs2,  v = agent(adv_clone)

    #hid_arr2.append(adv_clone.tolist())
    # print(inputs)
    #policy = probs.clone()
    s1=torch.tensor(0.0)
    s2 = torch.tensor(0.0)
    policy = probs2.clone()
    # print(policy)
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0
    lst2 = policy.tolist()[0]'''
    '''lst_temp=[]
    lst_temp2=[]
    for i in range(len(lst2)):
        if(lst2[i]==1 and lst[i]==0):
            lst_temp.append(i)
        elif(lst2[i]==1 and lst[i]==1):
            lst_temp2.append(i)'''
    '''f = open("block_details.csv", "a")
    writer = csv.writer(f)
    # print(energy)

    writer.writerow(lst_temp)
    writer.writerow(lst_temp2)
    f.close()'''
    '''tmp=[16,31,45]
    #for i in range(len(lst2)):
    for i in tmp:
        #if(lst2[i]==1 and lst[i]==0):
        s1=torch.add(s1,probs[0][i])
        s2=torch.add(s2, probs2[0][i])
    #print(s1)
    #print(s2)
    #time.sleep(5)
    start = time.time()
    s1.sum().backward()
    end = time.time()
    s2.sum().backward()'''
    #print(end - start)
    #time.sleep(500)
    #policy, sm2 = get_block_count(agent, new_input_adv)
    '''np.save(
        'img_adv_' + str(step) + '_' + str(sm) + '_' + str(final_sm) + '_data.npy',
        adv_img_min)'''
    #saliency, _ = torch.max(input_var.grad.data.abs(), dim=1)
    #saliency = saliency.reshape(32, 32).data.cpu().numpy()
    #np.save('sal_map/orig_sal2_new' + str(step) + '.npy', saliency)
    #print(input_var.grad.data)
    '''saliency = input_var.grad.data
    saliency = saliency.reshape(3, 32, 32).tolist()
    orig_sal.append(saliency)
    np.save('orig_sals_'  + '.npy', np.asarray(orig_sal))
    #print(torch.max(adv_clone.grad.data.abs(), dim=1))
    saliency = adv_clone.grad.data
    saliency = saliency.reshape(3, 32, 32).tolist()
    best_sal.append(saliency)
    np.save('best_sals_' + '.npy', np.asarray(best_sal))'''

    np.save('orig_C10_data_ILFO.npy',
            np.asarray(hid_arr))

    np.save('adv_C10_data_ILFO.npy',
            np.asarray(hid_arr2))
    '''np.save('ILFO/orig_bc_c10_test.npy',
            np.asarray(orig))

    np.save('ILFO/adv_bc_c10_test.npy',
            np.asarray(adv))'''
    np.save('orig_C100_hid_ILFO.npy',
            np.asarray(hid_vals))

    np.save('adv_C100_hid_ILFO_.npy',
            np.asarray(hid_vals_adv))

    #saliency, _ = torch.max(adv_clone.grad.data.abs(), dim=1)
    #best_saliency = saliency.reshape(32, 32).data.cpu().numpy()
    #np.save('sal_map/best_saliency2_new'+str(step)+'.npy', best_saliency)
    '''f = open("block_changed_whitebox_uni.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    blocks = []
    blocks.append(sm)
    blocks.append(sm2)
    writer.writerow(blocks)
    f.close()'''
    '''preds = rnet.forward_single(new_input_adv, new_policy.data.squeeze(0))
    softm=torch.softmax(preds,dim = 1)
    f = open("preds_uni.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    preds = preds.data.cpu().numpy()
    #blocks.append(sm2)
    writer.writerow(preds)
    f.close()
    np.save(
        'np_uni/' + str(step) +  '_'  + str(sm) + '_' + str(final_sm) + '_data.npy',
        adv_img_min)'''