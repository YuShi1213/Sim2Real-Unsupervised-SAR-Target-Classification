from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import proxynca
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import proxcy_mmd
import mmd
import adv
# Training settings
batch_size = 32
iteration=25000
lr = [0.001, 0.01]
momentum = 0.9
no_cuda =False
seed = 18
log_interval = 10
l2_decay = 5e-4
root_path = "/home/ai/Desktop/sim2real Unsupervised SAR Target Classification/datasets/"
src_name = "simplesimulate"
tgt_name = "matchv1"
tgt_name_test="simplereal_test"
tgt_pseudo_name = "simplereal"

cuda = not no_cuda and torch.cuda.is_available()
from torchvision import datasets, transforms
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

src_loader,_ = data_loader.load_training(root_path, src_name, batch_size, kwargs)
tgt_test_loader,_ = data_loader.load_training(root_path, tgt_name, batch_size, kwargs)
tgt_test_loader_test,_ = data_loader.load_testing(root_path, tgt_name_test, batch_size, kwargs)
tgt_pseudo_loader,tgt_pseudo_dataset = data_loader.load_training(root_path, tgt_pseudo_name, batch_size, kwargs)
src_dataset_len = len(src_loader.dataset)
src_loader_len = len(src_loader)
tgt_dataset_len = len(tgt_test_loader.dataset)
tgt_pseudo_dataset_len = len(tgt_pseudo_loader.dataset)
tgt_pseudo_loader_len = len(tgt_pseudo_loader)
tgt_loader_len = len(tgt_test_loader)
tgt_loader_len = len(tgt_test_loader)
tgt_dataset_test_len = len(tgt_test_loader_test.dataset)
tgt_test_loader_len = len(tgt_test_loader_test)

def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_test_loader)
    tgt_pseudo_iter = iter(tgt_pseudo_loader)
    
    correct = 0
    criterion = proxynca.ProxyNCA(nb_classes = 10,
    sz_embedding = 128).cuda()
    adv_grl = adv.AdversarialLoss().cuda()
    criterion_tgt = proxynca.ProxyNCA(nb_classes = 10,
    sz_embedding = 128).cuda()
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': lr[1]},
        {'params': model.cls_fc1.parameters(), 'lr': lr[1]},
        {'params': criterion.parameters(), 'lr': lr[0]},
        {'params': criterion_tgt.parameters(), 'lr': lr[0]},
        {'params': adv_grl.parameters(), 'lr': lr[0]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration+1):

        model.train()
       
        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

        try:
            src_data, src_label = src_iter.next()
            #tgt_data, tgt_label = tgt_iter.next()
            
        except Exception as err:
            src_iter=iter(src_loader)
            src_data, src_label = src_iter.next()
            #tgt_iter = iter(tgt_test_loader)
            #tgt_data, tgt_label = tgt_iter.next()
        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            #tgt_data, tgt_label = tgt_data.cuda(), tgt_label.cuda()
            #tgt_pseudo_data, tgt_pseudo_label = tgt_pseudo_data.cuda(),tgt_pseudo_label.cuda()
            

        optimizer.zero_grad()
        #all_data = torch.cat((src_data,tgt_data),dim=0)
        #all_label = torch.cat((src_label,tgt_label),dim=0)
        src_pred,src_ce,src_feat = model(src_data)
        
        
        src_loss, src_proxcy = criterion(src_pred, src_label.cuda(),None)
        src_ce_loss = F.nll_loss(F.log_softmax(src_ce, dim=1), src_label)
        #print(src_proxcy)
        
        if i <15000:
            try:
                tgt_data, tgt_label = tgt_iter.next()
            except Exception as err:
                tgt_iter = iter(tgt_test_loader)
                tgt_data, tgt_label = tgt_iter.next()
            if cuda:
                tgt_data, tgt_label = tgt_data.cuda(), tgt_label.cuda()
            tgt_pred,tgt_ce,tgt_feat = model(tgt_data)
            tgt_loss, tgt_proxcy = criterion_tgt(tgt_pred, tgt_label.cuda(),None)
            #class_dis = mmd.mmd_rbf_noaccelerate(src_proxcy,tgt_proxcy)
            class_dis = proxcy_dis(src_proxcy,tgt_proxcy)
            tgt_ce_loss = F.nll_loss(F.log_softmax(tgt_ce, dim=1), tgt_label)
            
            loss_all = tgt_loss + src_loss + 0.5*src_ce_loss + 0.5*tgt_ce_loss
            
            #loss_all = tgt_loss + src_loss 
        else:
            try:
                tgt_pseudo_data, _ = tgt_pseudo_iter.next()
            except Exception as err:
                tgt_pseudo_iter = iter(tgt_pseudo_loader)
                tgt_pseudo_data, _ = tgt_pseudo_iter.next()
            if cuda:
                tgt_pseudo_data = tgt_pseudo_data.cuda()
            #_, proxcy = criterion_tgt(tgt_pred, tgt_label.cuda())
            pseudo_CEloss,mask,tgt_pseudo_pred,pseudo_labels,tgt_pseudo_feat = pseudo(model,tgt_pseudo_data)
            pseudo_loss, tgt_proxcy = criterion_tgt(tgt_pseudo_pred, pseudo_labels.cuda(),mask)
            src_ins_loss,_ = criterion_tgt(src_pred, src_label.cuda(),None)
            pseudo_ins_loss, _ = criterion(tgt_pseudo_pred, pseudo_labels.cuda(),mask)
            class_dis = proxcy_dis_contra1(src_proxcy,tgt_proxcy)
            adv_loss = adv_grl(src_feat,tgt_pseudo_feat)
            loss_all = src_loss + 0.5*src_ce_loss+ pseudo_loss+0.5*pseudo_CEloss +0.5*adv_loss+0.5*class_dis    
        loss_all.backward()
        optimizer.step()
        
        if i % log_interval == 0:
            if i <15000:
                print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsrc_Loss: {:.6f}\ttgt_Loss: {:.6f}\tclass_dis_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss_all.item(),src_loss.item(),tgt_loss.item(),class_dis.item()))
            else:
                print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsrc_Loss: {:.6f}\ttgt_Loss: {:.6f}\tclass_dis_Loss: {:.6f}\tadv_loss:{:.6f}'.format(
                    i, 100. * i / iteration, loss_all.item(),src_loss.item(),tgt_loss.item(),class_dis.item(),adv_loss.item()))
            

        if i%(log_interval*20)==0:
            _, proxcy = criterion_tgt(src_pred, src_label.cuda(),None)
            t_correct = test(model,proxcy)
            if t_correct > correct:
                correct = t_correct
                torch.save(model.state_dict(),"./checkpoint/model.pth")
                torch.save(proxcy,"./checkpoint/proxcy.pth")
        if i == iteration:
                print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
              src_name, tgt_name, correct, 100. * correct / tgt_dataset_test_len ))
def proxcy_dis(source,target):
    sim = torch.cosine_similarity(source,target)
    dis = 1-sim
    loss = torch.mean(dis)
    return loss
def proxcy_dis_contra1(source,target):
    sim = torch.mm(source,target.t())
    dis = 1 - sim
    T = torch.eye(10).cuda()
    loss = torch.sum(-T * F.log_softmax(-dis, -1), -1)
    return loss.mean()
def pseudo(model, data):
    #model.train()
    threshold = 0.9
    #with torch.no_grad():
    tgt_pred_proxcy,tgt_pseudo_pred,tgt_pseudo_feat= model(data)
    prob = F.softmax(tgt_pseudo_pred, dim=1)
    max_probs,index = torch.max(prob,-1) 
    mask = max_probs.ge(threshold).float()  
    pred = tgt_pseudo_pred.data.max(1)[1] # get the index of the max log-probability
    loss = (F.nll_loss(F.log_softmax(tgt_pseudo_pred, dim = 1), pred.cuda(), reduction='none')*mask).mean()
    
    return loss,mask,tgt_pred_proxcy,pred,tgt_pseudo_feat

def test(model,proxcy):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    features = []
    target = []
    pred_all = []
    with torch.no_grad():
        bs_index = 0 
        for tgt_test_data, tgt_test_label in tgt_test_loader_test:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred,ce_pred,_= model(tgt_test_data)
            distances = torch.cdist(tgt_pred, proxcy)
            prob = F.softmax(-distances,-1).max(1)[0]
            
            pred = distances.data.min(1)[1] # get the index of the max log-probability
            ###change the labels of dataloader
            pred_np = np.array(pred.cpu())
            #tgt_test_loader.dataset.targets[0+32*bs_index:31+32*bs_index]= pred_np
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()     
    return correct
    
if __name__ == '__main__':
    model = models.DANNet(num_classes=10)
    print(model)
    if cuda:
        model.cuda()
    train(model)
    
