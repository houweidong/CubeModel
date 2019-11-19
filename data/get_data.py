import os
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, \
    RandomHorizontalFlip, Normalize, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
import torch.utils.data as data
from data.read_wider import WiderAttr
from data.read_berkeley import BerkeleyAttr
from data.read_newdata import NewdataAttr
from data.transforms import ToMaskedTargetTensor, ToMaskedTargetTensorPaper, \
    get_inference_transform_person, square_no_elastic


# Helper class for combining multiple (possibly heterogeneous) datasets into one pseudo dataset
class MultiDataset(data.Dataset):
    def __init__(self, datasets):
        assert isinstance(datasets, list)
        self.datasets = datasets

    def __getitem__(self, index):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            index -= len(d)

    def __len__(self):
        return sum([len(d) for d in self.datasets])


# Helper class for sub-sampling a given dataset, which can facilitate splitting one dataset into training/validation
class SubsampleDataset(data.Dataset):
    # indices is a shuffled list  [32, 11, 3, 5, ...], which represent the subsamples indexes  of the dataset
    def __init__(self, dataset, indices):
        assert isinstance(dataset, data.Dataset)
        self.dataset = dataset

        assert isinstance(indices, list) and max(indices) < len(self.dataset) and min(indices) >= 0
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


# Randomly split dataset into train/val subsets
def split_dataset_into_train_val(train_data, val_data, val_ratio=0.1):
    n_total = len(train_data)
    indices = list(range(n_total))

    # 100 sample  retio = 0.1  so split = 10
    split = int(np.floor(val_ratio * n_total))

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    return SubsampleDataset(train_data, train_idx), SubsampleDataset(val_data, valid_idx)


def _get_widerattr(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'Wider')
    cropping_transform = get_inference_transform_person
    train_img_transform = Compose(
        [square_no_elastic, RandomHorizontalFlip(), RandomRotation(10, expand=True),
         # [RandomHorizontalFlip(), RandomRotation(10, expand=True),
         Resize((opt.person_size, opt.person_size)),
         ToTensor(), Normalize(mean, std)])
    # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    val_img_transform = Compose(
        [square_no_elastic,
         Resize((opt.person_size, opt.person_size)),
         ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs) if opt.mode == 'branch' else ToMaskedTargetTensorPaper(attrs)

    train_data = WiderAttr(attrs, root, 'train', opt.mode, cropping_transform, img_transform=train_img_transform,
                           target_transform=target_transform)
    val_data = WiderAttr(attrs, root, 'test', opt.mode, cropping_transform,
                         img_transform=val_img_transform, target_transform=target_transform)

    return train_data, val_data


def _get_newdata(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'new')
    cropping_transform = get_inference_transform_person
    train_img_transform = Compose(
        [square_no_elastic, RandomHorizontalFlip(), RandomRotation(10, expand=True),
         # [RandomHorizontalFlip(), RandomRotation(10, expand=True),
         Resize((opt.person_size, opt.person_size)),
         ToTensor(), Normalize(mean, std)])
    # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    val_img_transform = Compose(
        [square_no_elastic,
         Resize((opt.person_size, opt.person_size)),
         ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs)

    train_data = NewdataAttr(attrs, root, 'train', opt.mode, cropping_transform, img_transform=train_img_transform,
                             target_transform=target_transform)
    val_data = NewdataAttr(attrs, root, 'test', opt.mode, cropping_transform,
                           img_transform=val_img_transform, target_transform=target_transform)

    return train_data, val_data


def _get_berkeley(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'attributes_dataset')
    cropping_transform = get_inference_transform_person
    train_img_transform = Compose(
        [square_no_elastic, RandomHorizontalFlip(), RandomRotation(10, expand=True),
         # [RandomHorizontalFlip(), RandomRotation(10, expand=True),
         Resize((opt.person_size, opt.person_size)), ToTensor(), Normalize(mean, std)])
    # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    val_img_transform = Compose(
        [square_no_elastic, Resize((opt.person_size, opt.person_size)), ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensorPaper(attrs)

    train_data = BerkeleyAttr(attrs, root, 'train', opt.mode, cropping_transform, img_transform=train_img_transform,
                              target_transform=target_transform)
    val_data = BerkeleyAttr(attrs, root, 'test', opt.mode, cropping_transform,
                            img_transform=val_img_transform, target_transform=target_transform)

    return train_data, val_data


_dataset_getters = {'Wider': _get_widerattr, 'Berkeley': _get_berkeley, 'New': _get_newdata}


def get_data(opt, available_attrs, mean, std):
    names = opt.dataset.split(",")

    # Get and collect each dataset which will be combined later
    train_data = []
    val_data = []
    for name in names:
        assert name in _dataset_getters
        train, val = _dataset_getters[name](opt, mean, std, available_attrs)
        train_data.append(train)
        val_data.append(val)

    # Combine multiple datasets if necessary
    if len(names) > 1:
        train_data = MultiDataset(train_data)
        val_data = MultiDataset(val_data)
    else:
        train_data = train_data[0]
        val_data = val_data[0]

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.n_threads, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.n_threads,
                            pin_memory=True)

    return train_loader, val_loader
