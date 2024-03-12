from __future__ import print_function
import argparse
import torch
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
import adv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
# Training settings
batch_size = 16
iteration=30000
lr = [0.001, 0.01]
momentum = 0.9
no_cuda =False
seed = 18
log_interval = 10
l2_decay = 5e-4
root_path = "/media/ai/d899633d-2ef9-4e8d-ac69-eb22cae4d04f/gyc/shiyu/mstar/deep-transfer-learning-master/UDA/pytorch1.0/DAN/dataset/inductive/"
src_name = "simple_all"
tgt_name = "simplereal"
tgt_name_test="simplereal_test"
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

src_loader = data_loader.load_training(root_path, src_name, batch_size, kwargs)
#tgt_train_loader = data_loader.load_training(root_path, tgt_name, batch_size, kwargs)
tgt_test_loader = data_loader.load_testing(root_path, tgt_name, batch_size, kwargs)
tgt_test_loader_test = data_loader.load_testing(root_path, tgt_name_test, batch_size, kwargs)
src_dataset_len = len(src_loader.dataset)
tgt_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_test_loader)
tgt_dataset_test_len = len(tgt_test_loader_test.dataset)
tgt_test_loader_len = len(tgt_test_loader_test)
def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_test_loader)
    correct = 0
    criterion = proxynca.ProxyNCA(nb_classes = 10,
    sz_embedding = 128).cuda()
    adv_grl = adv.AdversarialLoss().cuda()
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)
    for i in range(1, iteration+1):
        model.train()
       
        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter=iter(src_loader)
            src_data, src_label = src_iter.next()
            
        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

        optimizer.zero_grad()
        src_pred = model(src_data)
        loss, _ = criterion(src_pred, src_label.cuda())
        #cls_loss = F.nll_loss(F.log_softmax(src_pred_CE, dim=1), src_label)
        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1
        #loss_all = loss+0.5*cls_loss
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\t'.format(
                i, 100. * i / iteration, loss.item()))

        if i%(log_interval*20)==0:
            _, proxcy = criterion(src_pred, src_label.cuda())
            t_correct = test(model,proxcy)
            if t_correct > correct:
                correct = t_correct
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
              src_name, tgt_name, correct, 100. * correct / tgt_dataset_test_len ))
        
def test(model,proxcy):
    model.eval()
    test_loss = 0
    correct = 0
    features = []
    target = []
    pred_all = []
    proxcy_label = np.array([10,11,12,13,14,15,16,17,18,19])

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in tgt_test_loader_test:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred= model(tgt_test_data)
            distances = torch.cdist(tgt_pred, proxcy)
            #test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim = 1), tgt_test_label, reduction='sum').item() # sum up batch loss
            pred = distances.data.min(1)[1] # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()
            outputs = tgt_pred.view(-1, 128)
            outputs = outputs.data.cpu().numpy()
            features.append(outputs)
            labels = np.array(tgt_test_label.cpu())
            target.append(labels)
        array_proxcy = proxcy.data.cpu().numpy()
        features.append(array_proxcy)
        features = np.concatenate(features,axis=0)
        target.append(proxcy_label)
        label_final_use = np.concatenate(target)
    tsne = TSNE(n_components=2, init='pca', random_state=1)
    fea_tsne = tsne.fit_transform(features)
    plot_tsne(fea_tsne, label_final_use, 1)
    #test_loss /= tgt_dataset_test_len
    print(' Accuracy: {}/{} ({:.2f}%)\n'.format(
    correct, tgt_dataset_test_len,
        100. * correct / tgt_dataset_test_len))
    return correct

def plot_tsne(fea_tsne, label, task):
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
    path = os.path.join("task_{}tsne.png".format(task))
    plt.savefig(path)

if __name__ == '__main__':
    model = models.DANNet(num_classes=10)
    print(model)
    if cuda:
        model.cuda()
    train(model)
    
