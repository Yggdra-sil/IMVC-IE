from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py

class my(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path)['X1'].transpose().astype(np.float32)
        data2 = scipy.io.loadmat(path)['X2'].transpose().astype(np.float32)
        labels = scipy.io.loadmat(path)['gt']
        self.V1 = data1
        self.V2 = data2
        self.Y = labels

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
           self.V2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):
        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class myh5py(Dataset):
    def __init__(self, path):
        data1 = np.array(h5py.File(path)['X1']).astype(np.float32)
        data2 = np.array(h5py.File(path)['X2']).astype(np.float32)
        labels = np.array(h5py.File(path)['gt']).transpose().astype(np.float32)
        self.V1 = data1
        self.V2 = data2
        self.Y = labels

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
           self.V2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):

        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.V1 = data1
        self.V2 = data2
        self.Y = labels

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
           self.V2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):
        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)            
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def percentage_dele(self, road, avail1, avail2, pair):
        if road == 2:
            train_sample1 = []
            train_sample2 = []
            for i in pair:
                train_sample1.append(self.V1[i])
                train_sample2.append(self.V2[i])
            train_sample1 = np.array(train_sample1).astype(np.float32)
            train_sample2 = np.array(train_sample2).astype(np.float32)
            self.V1 = train_sample1
            self.V2 = train_sample2

        elif road == 1:
            avail_sample1 = []
            avail_sample2 = []
            for i in avail1:
                avail_sample1.append(self.V1[i])
            for i in avail2:
                avail_sample2.append(self.V2[i])
            avail_sample1 = np.array(avail_sample1).astype(np.float32)
            avail_sample2 = np.array(avail_sample2).astype(np.float32)
            self.V1 = avail_sample1
            self.V2 = avail_sample2

    def sample_mean(self):
        V1_mean = self.V1.mean(axis=0).astype(np.float32)
        V2_mean = self.V2.mean(axis=0).astype(np.float32)
        return V1_mean, V2_mean

    def pretrain_sigma(self):
        view1 = self.V1.astype(np.float32)
        view2 = self.V2.astype(np.float32)
        for i in range(view1.shape[0]):
            view1[i] = (view1[i] - view1[i].min()) / (view1[i].max() - view1[i].min())
            view2[i] = (view2[i] - view2[i].min()) / (view2[i].max() - view2[i].min())
        sigma1 = view1.std(axis=0)
        sigma2 = view2.std(axis=0)
        return sigma1, sigma2


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST_USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "my_UCI":
        dataset = my(('./data/' + dataset + '.mat'))
        dims = [240, 76]
        view = 2
        class_num = 10
        data_size = 2000
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
