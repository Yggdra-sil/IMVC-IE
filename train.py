import torch
from network import Network
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from scipy import stats
import h5py
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MNIST_USPS
# BDGP
# my_UCI

# Dataname = 'MNIST_USPS'
Dataname = 'BDGP'
# Dataname = 'my_UCI'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--temperature_f", default=0.1)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=50)
parser.add_argument("--con_epochs", default=200)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)

dataset_pretrain = copy.deepcopy(dataset)
dataset_contrain = copy.deepcopy(dataset)


tot_sample = dataset.V1.shape[0]
p = 0.5
np.random.seed(10)

miss_mark1 = np.random.choice(tot_sample, size=int(tot_sample * p), replace=False)
miss_mark1.sort()
available_mark1 = []
for i in range(tot_sample):
    if i not in miss_mark1:
        available_mark1.append(i)
available_mark1 = np.array(available_mark1)

miss_mark2 = np.random.choice(tot_sample, size=int(tot_sample * p), replace=False)
miss_mark2.sort()
available_mark2 = []
for i in range(tot_sample):
    if i not in miss_mark2:
        available_mark2.append(i)
available_mark2 = np.array(available_mark2)

pair_mark = []
for i in available_mark1:
    if i in available_mark2:
        pair_mark.append(i)
pair_mark = np.array(pair_mark)
print("pair_mark: ",len(pair_mark))

dataset_pretrain.percentage_dele(1, available_mark1, available_mark2, pair_mark)
dataset_contrain.percentage_dele(2, available_mark1, available_mark2, pair_mark)


data_loader_pretrain = torch.utils.data.DataLoader(
    dataset_pretrain,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
data_loader_contrain = torch.utils.data.DataLoader(
    dataset_contrain,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True



def pretrain(epoch):
    tot_loss = 0.
    pretrain_criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader_pretrain):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(pretrain_criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader_pretrain)))


def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader_contrain):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    if len(data_loader_contrain) == 0:
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss))
    else:
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader_contrain)))


def generate_prompt_box(view_mean, mu, sigma, number):
    view_mean = view_mean.reshape(1, -1)
    vm_nor = (view_mean - np.min(view_mean)) / (np.max(view_mean) - np.min(view_mean))                      
    simulated_box = []
    for i in range(number):
        noise = np.random.normal(mu, sigma, view_mean.shape)
        simulated_sample = vm_nor + noise
        simulated_sample = simulated_sample * (np.max(view_mean) - np.min(view_mean)) + np.min(view_mean)   
        simulated_sample = simulated_sample.reshape(view_mean.shape[1])
        simulated_box.append(simulated_sample)
    simulated_box = np.array(simulated_box).astype(np.float32)
    simulated_box = torch.from_numpy(simulated_box)
    return simulated_box


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)



if not os.path.exists('./models'):
    os.makedirs('./models')
seed = 10
T = 1
for i in range(T):
    print(Dataname)
    print("ROUND:{}".format(i + 1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)
        epoch += 1



view1_mean, view2_mean = dataset_pretrain.sample_mean()
r = 1
mu = 0
tot_sample1_pretrain = len(available_mark1)
tot_sample2_pretrain = len(available_mark2)
sigma1, sigma2 = dataset_pretrain.pretrain_sigma()
sigma1 = sigma1.reshape(1, -1)
sigma2 = sigma2.reshape(1, -1)
dataset_rec = copy.deepcopy(dataset)
dataset_rec_noise = copy.deepcopy(dataset)
dataset_rec_NOE = copy.deepcopy(dataset)
dataset_rec_mean = copy.deepcopy(dataset)
dataset_rec_zero = copy.deepcopy(dataset)
count = 0
for i in miss_mark1:
    view2_correspond = (torch.from_numpy(dataset.V2[i]).reshape(1, -1)).to(device)                              
    prompt_box = generate_prompt_box(view1_mean, mu, sigma1, number=int(r * tot_sample1_pretrain)).to(device)   
    vp = [prompt_box, view2_correspond]   
    hs, _, _, _ = model(vp)  
       
    similarity = torch.nn.CosineSimilarity(dim=1)
    sim = similarity(hs[1], hs[0])
    sim_value, sim_mark = torch.topk(sim, k=1)      
    best_simulated_cpu = prompt_box[sim_mark].cpu().numpy().reshape(view1_mean.shape).astype(np.float32)       
    dataset_rec.V1[i] = best_simulated_cpu

print()
for i in miss_mark2:
    view1_correspond = (torch.from_numpy(dataset.V1[i]).reshape(1, -1)).to(device)
    prompt_box = generate_prompt_box(view2_mean, mu, sigma2, number=int(r * tot_sample1_pretrain)).to(device)
    vp = [view1_correspond, prompt_box]
    hs, _, _, _ = model(vp)

    similarity = torch.nn.CosineSimilarity(dim=1)
    sim = similarity(hs[0], hs[1])
    sim_value, sim_mark = torch.topk(sim, k=1)
    best_simulated_cpu = prompt_box[sim_mark].cpu().numpy().reshape(view2_mean.shape).astype(np.float32)
    dataset_rec.V2[i] = best_simulated_cpu


data_loader_st = torch.utils.data.DataLoader(
     dataset_rec,
     batch_size=args.batch_size,
     shuffle=True,
     drop_last=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def semantic_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader_st):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader_st)))

while epoch <= args.mse_epochs + args.con_epochs + 100:
    semantic_train(epoch)
    epoch += 1

rec_v1 = torch.from_numpy(dataset_rec.V1).reshape(data_size, -1).to(device)
rec_v2 = torch.from_numpy(dataset_rec.V2).reshape(data_size, -1).to(device)
_, qs, _, _ = model([rec_v1, rec_v2])
confi1, y_pred1 = qs[0].topk(k=1, dim=1)
y_pred1 = y_pred1.cpu().numpy()
confi2, y_pred2 = qs[1].topk(k=1, dim=1)
y_pred2 = y_pred2.cpu().numpy()

pur1 = purity_score(dataset.Y.reshape(data_size), y_pred1.reshape(data_size))
pur2 = purity_score(dataset.Y.reshape(data_size), y_pred2.reshape(data_size))
print('semantic_V1_acc: ', cluster_acc(dataset.Y, y_pred1))
print('semantic_V2_acc: ', cluster_acc(dataset.Y, y_pred2))
print("semantic_V1_pur1: ", pur1)
print("semantic_V2_pur2: ", pur2)

