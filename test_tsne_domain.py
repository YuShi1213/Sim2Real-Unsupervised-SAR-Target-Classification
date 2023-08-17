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
import random
root_path = "/media/ai/data/shiyu/mstar/deep-transfer-learning-master/UDA/pytorch1.0/DAN/dataset/inductive/"
src_name_test = "simplesimulate_test"
tgt_name_test="simplereal_test"
checkpoint_path = "/media/ai/data/shiyu/mstar/deep-transfer-learning-master/UDA/pytorch1.0/Proxcy_UDA_adv_instance/checkpoint/model.pth"
proxcy_path = "/media/ai/data/shiyu/mstar/deep-transfer-learning-master/UDA/pytorch1.0/Proxcy_UDA_adv_instance/checkpoint/proxcy.pth"
batch_size = 32
cuda = torch.cuda.is_available()
from torchvision import datasets, transforms
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

src_test_loader_test,_= data_loader.load_testing(root_path, src_name_test, batch_size, kwargs)
src_dataset_test_len = len(src_test_loader_test.dataset)
src_test_loader_len = len(src_test_loader_test)

tgt_test_loader_test,_ = data_loader.load_testing(root_path, tgt_name_test, batch_size, kwargs)
tgt_dataset_test_len = len(tgt_test_loader_test.dataset)
tgt_test_loader_len = len(tgt_test_loader_test)


def test(model,proxcy):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    features = []
    target = []
    pred_all = []
    with torch.no_grad():
        
        for tgt_test_data, tgt_test_label in tgt_test_loader_test:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred,ce_pred,pred_2048 = model(tgt_test_data)
            distances = torch.cdist(tgt_pred, proxcy)
            prob = F.softmax(-distances,-1).max(1)[0]
            #test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim = 1), tgt_test_label, reduction='sum').item() # sum up batch loss
            pred = distances.data.min(1)[1] # get the index of the max log-probability
            ###change the labels of dataloader
            pred_np = np.array(pred.cpu())
            #tgt_test_loader.dataset.targets[0+32*bs_index:31+32*bs_index]= pred_np
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()
            #outputs = tgt_pred.view(-1, 128)
            #outputs = outputs.data.cpu().numpy()
            outputs = pred_2048.data.cpu().numpy()
            features.append(outputs)
        for src_test_data, src_test_label in src_test_loader_test:
            if cuda:
                src_test_data, src_test_label = src_test_data.cuda(), src_test_label.cuda()
            src_test_data, src_test_label = Variable(src_test_data), Variable(src_test_label)
            
            src_pred,ce_pred,pred_2048_s= model(src_test_data)
            
            #outputs = src_pred.view(-1, 128)
            #outputs = outputs.data.cpu().numpy()
            outputs = pred_2048_s.data.cpu().numpy()
            features.append(outputs)
        labels_t = np.ones(tgt_dataset_test_len)
        labels_s = np.zeros(src_dataset_test_len)
        features = np.concatenate(features,axis=0)
        target.append(labels_t)
        target.append(labels_s)
        label_final_use = np.concatenate(target)
    tsne = TSNE(n_components=2, init='pca', random_state=1)
    fea_tsne = tsne.fit_transform(features)
    plot_tsne(fea_tsne, label_final_use)          
    print(' Accuracy: {}/{} ({:.2f}%)\n'.format(
    correct, tgt_dataset_test_len,
        100. * correct / tgt_dataset_test_len))       
    return correct

def plot_tsne(fea_tsne, label):
    X_norm = fea_tsne
    num_class = len(np.unique(label))
    vis_x = X_norm[:, 0]
    vis_y = X_norm[:, 1]
    label_data = label

    plt.figure(figsize=(8, 8))
    plt.scatter(vis_x, vis_y, c=label_data, cmap=plt.cm.get_cmap("jet", num_class), marker='.')
    plt.colorbar(ticks=range(num_class))
    plt.xticks([])
    plt.yticks([])
    path = os.path.join("tsne_s_t_embding.png")
    plt.savefig(path)

if __name__ == '__main__':
    model = models.DANNet(num_classes=10)
    params = torch.load(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    proxcy = torch.load(proxcy_path)
    
    print(model)
    if cuda:
        model.cuda()
    test(model,proxcy)
    
