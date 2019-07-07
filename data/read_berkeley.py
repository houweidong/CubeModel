from torch.utils.data import Dataset
import os
from data.attributes import BerkeleyAttributes as BkAt, Attribute, AttributeType as AtTp
from torchvision.datasets.folder import pil_loader
# import json
from data.image_loader import opencv_loader


class BerkeleyAttr(Dataset):
    def __init__(self, attributes, root, subset, mode, cropping_transform,
                 img_transform=None, target_transform=None):
        for attr in attributes:
            assert isinstance(attr, Attribute)
        self._attrs = attributes
        # mode is in ["paper", "branch"]
        self.mode = mode
        self.data = self._make_dataset(root, subset)

        self.cropping_transform = cropping_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.img_loader = opencv_loader

    def _make_dataset(self, root, subset):
        assert subset in ['train', 'val', 'test']

        data = []

        if subset in ['train']:
            anno_file_dir = os.path.join(root, 'train')
        else:
            anno_file_dir = os.path.join(root, 'test')

        anno_file = os.path.join(anno_file_dir, 'labels.txt')

        with open(anno_file, 'r') as f:
            for line in f:
                img = list(filter(None, line.strip().split(' ')))
                sample = dict(img=os.path.join(anno_file_dir, img[0]),
                              bbox=list(map(float, img[1:5])))
                if img[1] == 'NaN' or img[2] == 'NaN' or img[3] == 'NaN' or img[4] == 'NaN':
                    continue
                recognizability = dict()
                if self.mode == 'paper' and subset == 'train':
                    # recognizability = dict()
                    for attr, index in zip(self._attrs, range(5, 14)):
                        attr = attr.key
                        if img[index] == '1':
                            sample[attr] = 1
                        else:
                            sample[attr] = 0
                        recognizability[attr] = 1
                    sample['recognizability'] = recognizability

                elif self.mode == 'branch' or (self.mode == 'paper' and subset != 'train'):
                    for attr, index in zip(self._attrs, range(5, 14)):
                        attr = attr.key
                        if img[index] != '0':
                            # -1 => 0  1=> 1
                            sample[attr] = (int(img[index]) + 1) // 2
                            recognizability[attr] = 1
                        else:  # Attribute is unrecognizable
                            # Treat attribute is available only if recognizability is considered
                            sample[attr] = -10  # Dummy value
                            recognizability[attr] = 0
                    # In this dataset every attribute may be unrecognizable
                    sample['recognizability'] = recognizability
                else:
                    raise Exception('not supported mode, where the mode come from [{}]'.format(self.mode))
                data.append(sample)

        return data

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample['img']
        bbox = sample['bbox']
        img = self.img_loader(img_path)

        crop = self.cropping_transform((img, bbox))

        # Transform image crop
        if self.img_transform is not None:
            crop = self.img_transform(crop)

        # Transform target
        target = sample.copy()  # Copy sample so that the original one won't be modified
        target.pop('img')
        target.pop('bbox')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return crop, target

    def __len__(self):
        return len(self.data)
