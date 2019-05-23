import pretrainedmodels
from torchvision.models import vgg
import torch.nn as nn
from data.attributes import Attribute
import torch
from model import detnet, fpn18

def modify_vgg(model):
    model._features = model.features
    del model.features
    del model.classifier  # Delete unused module to free memory

    def features(self, input):
        x = self._features(input)
        return x

    # TODO Based on pretrainedmodels, it modify instance method instead of class. Will need to test.py
    setattr(model.__class__, 'features', features)  # Must use setattr here instead of assignment

    return model


def get_backbone_network(conv, pretrained=True):
    if conv.startswith('vgg'):  # For VGG, use torchvision directly instead
        vgg_getter = getattr(vgg, conv)
        backbone = vgg_getter(pretrained=pretrained)
        feature_map_depth = 512

        # Modify the VGG model to make it align with pretrainmodels' format
        backbone = modify_vgg(backbone)
    elif conv.startswith('detnet'): # just support detenet59 now
        detnet_getter = getattr(detnet, conv)
        backbone = detnet_getter(pretrained=pretrained)
        feature_map_depth = 1024
    elif conv.startswith('fpn'):
        fpn_getter = getattr(fpn18, conv)
        backbone = fpn_getter()
        feature_map_depth = backbone.out_channels
    else:
        if pretrained:
            backbone = pretrainedmodels.__dict__[conv](num_classes=1000)
        else:
            backbone = pretrainedmodels.__dict__[conv](num_classes=1000, pretrained=None)
        feature_map_depth = backbone.last_linear.in_features

    # use mean and std to do what
    if not hasattr(backbone, 'mean'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = backbone.mean, backbone.std

    return backbone, feature_map_depth, mean, std


def _generate_param_group(modules):
    for module in modules:
        for name, param in module.named_parameters():
            yield param


def get_param_groups(group_of_layers):
    assert isinstance(group_of_layers, list)
    param_groups = []
    for group in group_of_layers:
        param_groups.append({'params': _generate_param_group(group)})
    return param_groups


class Base(nn.Module):
    def __init__(self, attributes, in_features, out_features, img_size):
        for attr in attributes:
            assert isinstance(attr, Attribute)
        super(Base, self).__init__()
        self.img_size = img_size
        # self.map_size = int(self.img_size / 224 * 7)
        self.in_features = in_features
        self.out_features = out_features
        self.attributes = attributes
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        for attr in self.attributes:
            name = attr.name
            setattr(self, 'fc_' + name + '_classifier', nn.Linear(512, 2))
            # Also define a branch for classifying recognizability if necessary
            if attr.rec_trainable:
                setattr(self, 'fc_' + name + '_recognizable', nn.Linear(512, 2))
            # setattr(self, 'fc_' + name + '_1', nn.Linear(self.in_features, 512))

    def forward(self, *parameters):
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.

        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        raise NotImplementedError


class NoAttention(Base):
    def __init__(self, attributes, in_features, out_features, norm_size):
        super(NoAttention, self).__init__(attributes, in_features, out_features, norm_size[1])

        # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        for attr in self.attributes:
            setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))

    def forward(self, x):
        x = self.global_pool(x).view(x.size(0), -1)
        results = []
        for attr in self.attributes:
            name = attr.name
            y = getattr(self, 'fc_' + name + '_1')(x)
            y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        return results


class ScOd(Base):
    def __init__(self, attributes, in_features, out_features, norm_size):
        super(ScOd, self).__init__(attributes, in_features, out_features, norm_size[1])
        # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = norm_size[0]
        # Also define attention layer for attribute
        for attr in self.attributes:
            name = attr.name
            setattr(self, 'attention_' + name + '_xb', nn.Conv2d(self.in_features, 1, (1, 1)))
            setattr(self, 'attention_' + name + '_xa', nn.Conv2d(self.in_features, 512, (1, 1)))
            # if self.norm:
            #     setattr(self, 'att_nl', nn.Softmax(2))

    def forward(self, x):
        results = []
        for attr in self.attributes:
            name = attr.name
            xb = getattr(self, 'attention_' + name + '_xb')(x)
            # print(xb.shape)
            xa = getattr(self, 'attention_' + name + '_xa')(x)
            # print(xa.shape)
            xa = self.sigmoid(xa) if self.norm else xa
            # .sum((2, 3)) sum func has big problem, which make pro
            # collapse when GPU memory has half empty
            y = xb.mul(xa)
            y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
            y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        return results


class OvFc(NoAttention):
    def __init__(self, attributes, in_features, out_features, norm_size):
        super(OvFc, self).__init__(attributes, in_features, out_features, norm_size)
        self.norm = norm_size[0]
        # Also define attention layer for attribute
        for attr in self.attributes:
            name = attr.name
            setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (1, 1)))
            setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (1, 1)))
            setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 1, (1, 1)))

    def forward(self, x):
        results = []
        for attr in self.attributes:
            name = attr.name
            cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
            cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))
            cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
            cv3_rl = self.sigmoid(cv3_rl) if self.norm else cv3_rl
            y = cv3_rl * x
            y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
            y = getattr(self, 'fc_' + name + '_1')(y)
            y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        return results


class PrTp(NoAttention):
    def __init__(self, attributes, in_features, out_features, norm_size):
        super(PrTp, self).__init__(attributes, in_features, out_features, norm_size)
        self.norm = norm_size[0]
        # Also define attention layer for attribute
        for attr in self.attributes:
            name = attr.name
            # 10 prototype just for test
            setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
            setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (3, 3), padding=1))
            setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 10, (1, 1)))

            # setattr(self, 'prototype_' + name + '_coe1', nn.AdaptiveAvgPool2d(1))
            setattr(self, 'prototype_' + name + '_coe1', nn.Linear(self.in_features, int(self.in_features / 16)))
            setattr(self, 'prototype_' + name + '_coe2', nn.Linear(int(self.in_features / 16), 10))
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        results = []
        for attr in self.attributes:
            name = attr.name
            cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
            cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))
            cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
            cv3_rl = self.sigmoid(cv3_rl) if self.norm else cv3_rl
            # cv3_rl = self.relu(cv3_rl) if self.norm else cv3_rl

            # compute prototype coefficient
            # prototype_coe1 = self.relu(getattr(self, 'prototype_' + name + '_coe1')(
            # self.global_pool(x).view(x.size(0), -1)))
            # prototype_coe2 = self.sigmoid(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))
            prototype_coe1 = getattr(self, 'prototype_' + name + '_coe1')(self.global_pool(x).view(x.size(0), -1))
            prototype_coe2 = self.tanh(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))

            # multi prototype with attention map to produce new attention map
            new_attention = (prototype_coe2[..., None, None] * cv3_rl).sum(1, keepdim=True)
            y = new_attention * x
            y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
            y = getattr(self, 'fc_' + name + '_1')(y)
            y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        return results


class CamOvFc(NoAttention):
    def __init__(self, attributes, in_features, out_features, norm_size):
        super(CamOvFc, self).__init__(attributes, in_features, out_features, norm_size)
        self.norm = norm_size[0]
        # Also define attention layer for attribute
        for attr in self.attributes:
            name = attr.name
            # 10 prototype just for test
            # setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
            # setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (3, 3), padding=1))
            # setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 1, (1, 1)))
            setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (1, 1)))
            setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (1, 1)))
            setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 1, (1, 1)))
        self.softmax = torch.nn.Softmax(dim=2)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        results = []
        cam_results = []
        for attr in self.attributes:
            name = attr.name
            cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
            cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))
            cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
            # cv3_rl_sigmoid = self.sigmoid(cv3_rl) if self.norm else cv3_rl
            # cv3_rl = self.relu(cv3_rl) if self.norm else cv3_rl

            # compute softmax to cam
            cam_softmax = self.softmax(cv3_rl.view(cv3_rl.size(0), cv3_rl.size(1), -1))
            # cam_pool = self.pool(cam_softmax.view(cv3_rl.size()))
            cam = self.global_max_pool(cam_softmax)

            # y = cv3_rl_sigmoid * x
            y = cv3_rl + x
            y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
            y = getattr(self, 'fc_' + name + '_1')(y)
            y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
            # results.append(cam)
            cam_results.append(cam)
        results.extend(cam_results)
        return results
