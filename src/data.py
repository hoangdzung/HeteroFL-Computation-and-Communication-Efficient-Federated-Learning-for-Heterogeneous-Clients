import torch
import datasets
import numpy as np
from config import cfg
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import collections
import medmnist_class
import os 

def fetch_dataset(data_name, subset):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if not os.path.isdir(root):
        os.makedirs(root)
    if data_name == 'MNIST':
        dataset['train'] = datasets.MNIST(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        dataset['test'] = datasets.MNIST(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    elif data_name == 'CIFAR10':
        dataset['train'] = datasets.CIFAR10(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataset['test'] = datasets.CIFAR10(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif data_name in ['PennTreebank', 'WikiText2', 'WikiText103']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    elif data_name in medmnist_class.medmnist_classes :
        dataset['train'] = medmnist_class.medmnist_classes[data_name](root=root, split='train', download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])]))
        dataset['train'].target = dataset['train'].labels.squeeze().tolist()

        dataset['val'] = medmnist_class.medmnist_classes[data_name](root=root, split='train', download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])]))
        dataset['val'].target = dataset['val'].labels.squeeze().tolist()

        dataset['test'] = medmnist_class.medmnist_classes[data_name](root=root, split='test', download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])]))
        dataset['test'].target = dataset['test'].labels.squeeze().tolist()

    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], label_split = iid(dataset['train'], num_users)
        data_split['test'], _ = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'], label_split = non_iid(dataset['train'], num_users)
        data_split['test'], _ = non_iid(dataset['test'], num_users, label_split)
    else:
        raise ValueError('Not valid data split mode')
    return data_split, label_split


def iid(dataset, num_users):
    if cfg['data_name'] in ['MNIST', 'CIFAR10'] or cfg['data_name'] in medmnist_class.medmnist_classes:
        label = torch.tensor(dataset.target)
    elif cfg['data_name'] in ['WikiText2']:
        label = dataset.token
    else:
        raise ValueError('Not valid data name')

    d_idxs = np.random.permutation(len(dataset))
    local_datas = np.array_split(d_idxs, num_users)
    data_split, label_split = {}, {}

    # num_items = int(len(dataset) / num_users)
    # data_split, idx = {}, list(range(len(dataset)))
    # label_split = {}
    for i in range(num_users):
        data_split[i] = local_datas[i].tolist()
        label_split[i] = torch.unique(label[data_split[i]]).tolist()

    return data_split, label_split


def non_iid(dataset, num_users, label_split=None):
    label = np.array(dataset.target)
    cfg['non-iid-n'] = int(float(cfg['data_split_mode'].split('-')[-2]))
    dist = int(cfg['data_split_mode'].split('-')[-1])
    skew = float(cfg['data_split_mode'].split('-')[-2])

    K = len(set(label))
    # pair is (id, label)
    if cfg['data_name'] in medmnist_class.medmnist_classes:
        dpairs = [[did, dataset[did][1]] for did in range(len(dataset))]
    else:
        dpairs = [[did, dataset[did]['label'].item()] for did in range(len(dataset))]
    num = cfg['non-iid-n'] # each client contains only 'num' labels

    if dist==1:
        local_datas = [[] for _ in range(num_users)]
        if num == K:
            for k in range(K):
                # get list of ids which has label k
                idx_k = [p[0] for p in dpairs if p[1]==k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, num_users)
                for cid in range(num_users):
                    local_datas[cid].extend(split[cid].tolist())
        else:
            times = [0 for _ in range(num_users)]
            contain = []
            for i in range(num_users):
                current = [i % K] # set of label appear in client i
                times[i % K] += 1 # the total number of appearance of that label in all client
                j = 1 # the current size of the label set
                while (j < num):
                    ind = np.random.randint(0, K) # get a random label
                    if (ind not in current): # if label not in current label set of the client
                        j = j + 1 
                        current.append(ind) # add that label to the current label set
                        times[ind] += 1 
                contain.append(current)
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1]==k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[k])
                ids = 0
                # distribute subset of ids w.r.t the label to all clients having that label
                for cid in range(num_users):
                    if k in contain[cid]:
                        local_datas[cid].extend(split[ids].tolist())
                        ids += 1
    elif dist == 2:
        """label_skew_dirichlet"""
        min_size = 0
        local_datas = [[] for _ in range(num_users)]
        while min_size < 10:
            idx_batch = [[] for i in range(num_users)]
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1]==k]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(skew, num_users))
                ## Balance
                proportions = np.array([p * (len(idx_j) < len(dpairs)/ num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            local_datas[j].extend(idx_batch[j])
    elif dist ==4:
        MIN_ALPHA = 0.01
        alpha = (-4*np.log(skew + 10e-8))**4
        alpha = max(alpha, MIN_ALPHA)
        labels = [pair[-1] for pair in dpairs]
        lb_counter = collections.Counter(labels)
        p = np.array([1.0*v/len(dpairs) for v in lb_counter.values()])
        lb_dict = {}
        labels = np.array(labels)
        for lb in range(len(lb_counter.keys())):
            lb_dict[lb] = np.where(labels==lb)[0]
        proportions = [np.random.dirichlet(alpha*p) for _ in range(num_users)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(alpha * p) for _ in range(num_users)]
        while True:
            # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
            mean_prop = np.mean(proportions, axis=0)
            error_norm = ((mean_prop-p)**2).sum()
            print("Error: {:.8f}".format(error_norm))
            if error_norm<=1e-2/K:
                break
            exclude_norms = []
            for cid in range(num_users):
                mean_excid = (mean_prop*num_users-proportions[cid])/(num_users-1)
                error_excid = ((mean_excid-p)**2).sum()
                exclude_norms.append(error_excid)
            excid = np.argmin(exclude_norms)
            sup_prop = [np.random.dirichlet(alpha*p) for _ in range(num_users)]
            alter_norms = []
            for cid in range(num_users):
                if np.any(np.isnan(sup_prop[cid])):
                    continue
                mean_alter_cid = mean_prop - proportions[excid]/num_users + sup_prop[cid]/num_users
                error_alter = ((mean_alter_cid-p)**2).sum()
                alter_norms.append(error_alter)
            if len(alter_norms)>0:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]
        local_datas = [[] for _ in range(num_users)]
        # self.dirichlet_dist = [] # for efficiently visualizing
        for lb in lb_counter.keys():
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb] for pi in proportions])
            lb_proportion = lb_proportion/lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            # self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data+lb_data.tolist() for local_data,lb_data in zip(local_datas, lb_datas)]
        # self.dirichlet_dist = np.array(self.dirichlet_dist).T
        for i in range(num_users):
            np.random.shuffle(local_datas[i])        
    else:
        raise ValueError('Not valid dist')

    data_split, label_split={},{}
    for i in range(num_users):
        data_split[i] = local_datas[i]
        label_split[i] = list(set(label[data_split[i]]))

    return data_split, label_split


def make_data_loader(dataset):
    data_loader = {}
    for k in dataset:
        data_loader[k] = torch.utils.data.DataLoader(dataset=dataset[k], shuffle=cfg['shuffle'][k],
                                                     batch_size=cfg['batch_size'][k], pin_memory=True,
                                                     num_workers=cfg['num_workers'], collate_fn=input_collate)
    return data_loader


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        input = self.dataset[self.idx[index]]
        return input


class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[0]['label'].size(0)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        seq_length = min(self.seq_length, self.S - index)
        input = {'label': self.dataset[:]['label'][:, self.idx[index]:self.idx[index] + seq_length]}
        return input