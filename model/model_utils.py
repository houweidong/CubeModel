import pretrainedmodels
# from torchvision.models import vgg
import torch.nn as nn
from data.attributes import Attribute, AttributeType
import torch
from model import detnet, fpn18, MobileNetV2, mobilenetv3
from model import vgg
import math
import numpy as np
import torch.nn.functional as F

# class Scale(nn.Module):
#     def __init__(self, init_value=1.0):
#         super(Scale, self).__init__()
#         self.scale = nn.Parameter(torch.FloatTensor([init_value]))
#
#     def forward(self, input):
#         return input * self.scale


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


def modify_mobile(model):
    model._features = model.features
    del model.features
    del model.classifier  # Delete unused module to free memory

    def features(self, input):
        x = self._features(input)
        return x

    # TODO Based on pretrainedmodels, it modify instance method instead of class. Will need to test.py
    setattr(model.__class__, 'features', features)  # Must use setattr here instead of assignment

    return model

# for the old github owner mobile3
# def modify_mobilel(model):
#     del model.linear3
#     del model.bn3
#     del model.hs3
#     del model.linear4
#
#     def features(self, input):
#         out = self.hs1(self.bn1(self.conv1(input)))
#         out = self.bneck(out)
#         out = self.hs2(self.bn2(self.conv2(out)))
#         # out = F.avg_pool2d(out, 7)
#         # out = out.view(out.size(0), -1)
#         # out = self.hs3(self.bn3(self.linear3(out)))
#         # out = self.linear4(out)
#         return out
#
#     # TODO Based on pretrainedmodels, it modify instance method instead of class. Will need to test.py
#     setattr(model.__class__, 'features', features)  # Must use setattr here instead of assignment
#
#     return model
#
#
# def modify_mobiles(model):
#     del model.linear3
#     del model.bn3
#     del model.hs3
#     del model.linear4
#
#     def features(self, input):
#         out = self.hs1(self.bn1(self.conv1(input)))
#         out = self.bneck(out)
#         out = self.hs2(self.bn2(self.conv2(out)))
#         # out = F.avg_pool2d(out, 7)
#         # out = out.view(out.size(0), -1)
#         # out = self.hs3(self.bn3(self.linear3(out)))
#         # out = self.linear4(out)
#         return out
#
#     # TODO Based on pretrainedmodels, it modify instance method instead of class. Will need to test.py
#     setattr(model.__class__, 'features', features)  # Must use setattr here instead of assignment
#
#     return model


def modify_mobile3(model):
    model._features = model.features
    del model.avgpool
    del model.features
    del model.classifier  # Delete unused module to free memory

    def features(self, input):
        x = self._features(input)
        x = self.conv(x)
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
        feature_map_resol = 14

    elif conv.startswith('detnet'): # just support detenet59 now
        detnet_getter = getattr(detnet, conv)
        backbone = detnet_getter(pretrained=pretrained)
        feature_map_depth = 1024
        feature_map_resol = 14

    elif conv.startswith('fpn'):
        fpn_getter = getattr(fpn18, conv)
        backbone = fpn_getter(pretrained=pretrained)
        feature_map_depth = backbone.out_channels
        feature_map_resol = 14

    elif conv.startswith('mobilenet'):
        mobile_getter = getattr(MobileNetV2, conv)
        backbone = mobile_getter(pretrained=pretrained)
        feature_map_depth = backbone.last_channel

        backbone = modify_mobile(backbone)
        feature_map_resol = 14

    elif conv.startswith('mobile3l'):
        mobile_getter = getattr(mobilenetv3, conv)
        backbone = mobile_getter(pretrained=pretrained)
        feature_map_depth = 960

        backbone = modify_mobile3(backbone)
        feature_map_resol = 7

    elif conv.startswith('mobile3s'):
        mobile_getter = getattr(mobilenetv3, conv)
        backbone = mobile_getter(pretrained=pretrained)
        feature_map_depth = 576

        backbone = modify_mobile3(backbone)
        feature_map_resol = 7
    else:
        if pretrained:
            backbone = pretrainedmodels.__dict__[conv](num_classes=1000)
        else:
            backbone = pretrainedmodels.__dict__[conv](num_classes=1000, pretrained=None)
        feature_map_depth = backbone.last_linear.in_features
        feature_map_resol = 7

    # use mean and std to do what
    if not hasattr(backbone, 'mean'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = backbone.mean, backbone.std

    return backbone, feature_map_depth, feature_map_resol, mean, std


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
    def __init__(self, attributes, in_features, switch):
        for attr in attributes:
            assert isinstance(attr, Attribute)
        super(Base, self).__init__()
        # self.img_size = img_size
        # self.map_size = int(self.img_size / 224 * 7)
        self.in_features = in_features
        self.attributes = attributes
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.switch = switch

        for attr in self.attributes:
            name = attr.name
            setattr(self, 'fc_' + name + '_classifier', nn.Linear(512, 1))
            # Also define a branch for classifying recognizability if necessary
            if attr.rec_trainable:
                setattr(self, 'fc_' + name + '_recognizable', nn.Linear(512, 1))
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
    def __init__(self, attributes, in_features, dropout=0.1,
                 at=False, at_loss='mse', switch=True):
        super(NoAttention, self).__init__(attributes, in_features, switch)

        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        # self.global_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.global_pool = nn.AvgPool2d(7)
        for attr in self.attributes:
            setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))
        self.dropout = nn.Dropout(dropout)
        self.at = at
        self.at_loss = at_loss
        # self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch = x.size(0)
        ch = x.size(1)
        # resolu = x.size(2)
        # x = self.dropout(self.global_pool_1d(x.view(batch, ch, -1)).view(batch, -1))
        x = self.dropout(self.global_pool(x).view(x.size(0), -1))
        results = []
        for attr in self.attributes:
            name = attr.name
            y = self.dropout(getattr(self, 'fc_' + name + '_1')(x))
            # y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            if not self.switch:
                cls = self.sigmoid(cls)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                if not self.switch:
                    recognizable = self.sigmoid(recognizable)
                    results.append(recognizable)
        return results

        # y = getattr(self, 'fc_' + self.attributes[0].name + '_1')(x)
        # cls0 = getattr(self, 'fc_' + self.attributes[0].name + '_classifier')(y)
        # cls0 = self.sigmoid(cls0)

        # y = self.dropout(getattr(self, 'fc_' + self.attributes[1].name + '_1')(x))
        # cls1 = getattr(self, 'fc_' + self.attributes[1].name + '_classifier')(y)
        # cls1 = self.sigmoid(cls1)
        #
        # y = self.dropout(getattr(self, 'fc_' + self.attributes[2].name + '_1')(x))
        # cls2 = getattr(self, 'fc_' + self.attributes[2].name + '_classifier')(y)
        # cls2 = self.sigmoid(cls2)
        #
        # y = self.dropout(getattr(self, 'fc_' + self.attributes[3].name + '_1')(x))
        # cls3 = getattr(self, 'fc_' + self.attributes[3].name + '_classifier')(y)
        # cls3 = self.sigmoid(cls3)
        #
        # y = self.dropout(getattr(self, 'fc_' + self.attributes[4].name + '_1')(x))
        # cls4 = getattr(self, 'fc_' + self.attributes[4].name + '_classifier')(y)
        # cls4 = self.sigmoid(cls4)

        # return y


class OvFc(NoAttention):
    def __init__(self, attributes, in_features, dropout, at=False, at_loss='MSE'):
        super(OvFc, self).__init__(attributes, in_features, dropout, at, at_loss)
        for attr in self.attributes:
            name = attr.name
            setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
            setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 10, (3, 3), padding=1))
            setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(10, 1, (1, 1)))

    def forward(self, x):
        results = []
        results_at = []
        batch = x.size(0)
        ch = x.size(1)
        resolu = x.size(2)
        for attr in self.attributes:
            name = attr.name
            cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
            cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))
            cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
            if self.at:
                at = cv3_rl.view(batch, -1)
                at = self.log_softmax(at) if self.at_loss == 'KL' else at
                results_at.append(at)
                if self.at_loss == 'KL':
                    cv3_sm = self.softmax(cv3_rl.view(batch, -1))
                    # print(cv3_sm)
                    cv3_rl = cv3_sm * attr.at_coe
                    # print(cv3_rl[0, 14*7:14*8])
                    cv3_rl = cv3_rl.view(batch, 1, resolu, resolu)
            # if not self.at:
            #     cv3_rl = self.sigmoid(cv3_rl)
            # if name == 'maozi_yesno':
            #     print(cv3_rl[0, 0, :, 3:12])
            y = cv3_rl * x
            y = self.dropout(getattr(self, 'global_max_pool')(y).view(batch, -1))
            y = self.relu(y)
            y = self.dropout(getattr(self, 'fc_' + name + '_1')(y))
            y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        results.extend(results_at)
        return results


class PrTp(NoAttention):
    def __init__(self, attributes, in_features, dropout, at=False, at_loss='mse'):
        super(PrTp, self).__init__(attributes, in_features, dropout, at, at_loss)
        for attr in self.attributes:
            name = attr.name
            # 10 prototype just for test
            setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
            setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 10, (3, 3), padding=1))
            # setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 10, (1, 1)))

            # setattr(self, 'prototype_' + name + '_coe1', nn.AdaptiveAvgPool2d(1))
            setattr(self, 'prototype_' + name + '_coe1', nn.Linear(self.in_features, int(self.in_features / 16)))
            setattr(self, 'prototype_' + name + '_coe2', nn.Linear(int(self.in_features / 16), 10))

    def forward(self, x):
        results = []
        results_at = []
        batch = x.size(0)
        ch = x.size(1)
        resolu = x.size(2)
        for attr in self.attributes:
            name = attr.name
            cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
            cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))

            prototype_coe1 = getattr(self, 'prototype_' + name + '_coe1')(self.global_pool(x).view(x.size(0), -1))
            prototype_coe2 = self.tanh(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))

            # multi prototype with attention map to produce new attention map
            new_attention = (prototype_coe2[..., None, None] * cv2_rl).sum(1, keepdim=True)
            if self.at:
                at = new_attention.view(batch, -1)
                at = self.log_softmax(at) if self.at_loss == 'KL' else at
                results_at.append(at)
                if self.at_loss == 'KL':
                    new_attention = self.softmax(new_attention.view(batch, -1)) * attr.at_coe
                    new_attention = new_attention.view(batch, 1, resolu, resolu)
            y = new_attention * x
            y = self.dropout(getattr(self, 'global_pool')(y).view(y.size(0), -1))
            y = self.dropout(getattr(self, 'fc_' + name + '_1')(y))
            # y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        return results + results_at


class CPrTp(NoAttention):
    def __init__(self, attributes, in_features, dropout, at=False, at_loss='mse'):
        super(CPrTp, self).__init__(attributes, in_features, dropout, at, at_loss)

        setattr(self, 'attention_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
        setattr(self, 'attention_cv2', nn.Conv2d(512, 10, (3, 3), padding=1))

        for attr in self.attributes:
            name = attr.name
            setattr(self, 'prototype_' + name + '_coe1', nn.Linear(self.in_features, int(self.in_features / 16)))
            setattr(self, 'prototype_' + name + '_coe2', nn.Linear(int(self.in_features / 16), 10))

    def forward(self, x):
        results = []
        results_at = []
        batch = x.size(0)
        ch = x.size(1)
        resolu = x.size(2)
        cv1_rl = self.relu(getattr(self, 'attention_cv1')(x))
        cv2_rl = self.relu(getattr(self, 'attention_cv2')(cv1_rl))
        for attr in self.attributes:
            name = attr.name

            # cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
            # cv3_rl = self.sigmoid(cv3_rl)  #  if self.norm else cv3_rl
            # cv3_rl = self.relu(cv3_rl) if self.norm else cv3_rl

            # compute prototype coefficient
            # prototype_coe1 = self.relu(getattr(self, 'prototype_' + name + '_coe1')(
            # self.global_pool(x).view(x.size(0), -1)))
            # prototype_coe2 = self.sigmoid(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))
            prototype_coe1 = getattr(self, 'prototype_' + name + '_coe1')(self.global_pool(x).view(x.size(0), -1))
            prototype_coe2 = self.tanh(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))

            # multi prototype with attention map to produce new attention map
            new_attention = (prototype_coe2[..., None, None] * cv2_rl).sum(1, keepdim=True)
            if self.at:
                at = new_attention.view(batch, -1)
                at = self.log_softmax(at) if self.at_loss == 'KL' else at
                results_at.append(at)
                if self.at_loss == 'KL':
                    new_attention = self.softmax(new_attention.view(batch, -1)) * attr.at_coe
                    new_attention = new_attention.view(batch, 1, resolu, resolu)
            y = new_attention * x
            y = self.dropout(getattr(self, 'global_pool')(y).view(y.size(0), -1))
            y = self.dropout(getattr(self, 'fc_' + name + '_1')(y))
            # y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        return results + results_at


class PCPrTp(NoAttention):
    def __init__(self, attributes, in_features, dropout, at=False, at_loss='mse'):
        super(PCPrTp, self).__init__(attributes, in_features, dropout, at, at_loss)
        # self.norm = norm_size[0]
        # Also define attention layer for attribute
        # 10 prototype just for test
        setattr(self, 'attention_cv1', nn.Conv2d(self.in_features, 256, (3, 3), padding=1))
        setattr(self, 'attention_cv2', nn.Conv2d(256, 5, (3, 3), padding=1))

        for attr in self.attributes:
            name = attr.name
            setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 256, (3, 3), padding=1))
            setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(256, 5, (3, 3), padding=1))
            # setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 10, (1, 1)))

            # setattr(self, 'prototype_' + name + '_coe1', nn.AdaptiveAvgPool2d(1))
            setattr(self, 'prototype_' + name + '_coe1', nn.Linear(self.in_features, int(self.in_features / 16)))
            setattr(self, 'prototype_' + name + '_coe2', nn.Linear(int(self.in_features / 16), 10))

    def forward(self, x):
        results = []
        results_at = []
        batch = x.size(0)
        ch = x.size(1)
        resolu = x.size(2)
        cv1_rl_cm = self.relu(getattr(self, 'attention_cv1')(x))
        cv2_rl_cm = self.relu(getattr(self, 'attention_cv2')(cv1_rl_cm))
        for attr in self.attributes:
            name = attr.name
            cv1_rl_at = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
            cv2_rl_at = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl_at))
            cv2_rl = torch.cat([cv2_rl_cm, cv2_rl_at], 1)
            prototype_coe1 = getattr(self, 'prototype_' + name + '_coe1')(self.global_pool(x).view(x.size(0), -1))
            prototype_coe2 = self.tanh(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))

            # multi prototype with attention map to produce new attention map
            new_attention = (prototype_coe2[..., None, None] * cv2_rl).sum(1, keepdim=True)
            if self.at:
                at = new_attention.view(batch, -1)
                at = self.log_softmax(at) if self.at_loss == 'KL' else at
                results_at.append(at)
                if self.at_loss == 'KL':
                    new_attention = self.softmax(new_attention.view(batch, -1)) * attr.at_coe
                    new_attention = new_attention.view(batch, 1, resolu, resolu)
            y = new_attention * x
            y = self.dropout(getattr(self, 'global_pool')(y).view(y.size(0), -1))
            y = self.dropout(getattr(self, 'fc_' + name + '_1')(y))
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            results.append(cls)
            if attr.rec_trainable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                results.append(recognizable)
        return results + results_at


# class ScOd(Base):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(ScOd, self).__init__(attributes, in_features, out_features, norm_size[1])
#         # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.norm = norm_size[0]
#         # Also define attention layer for attribute
#         for attr in self.attributes:
#             name = attr.name
#             setattr(self, 'attention_' + name + '_xb', nn.Conv2d(self.in_features, 1, (1, 1)))
#             setattr(self, 'attention_' + name + '_xa', nn.Conv2d(self.in_features, 512, (1, 1)))
#             # if self.norm:
#             #     setattr(self, 'att_nl', nn.Softmax(2))
#
#     def forward(self, x):
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             xb = getattr(self, 'attention_' + name + '_xb')(x)
#             # print(xb.shape)
#             xa = getattr(self, 'attention_' + name + '_xa')(x)
#             # print(xa.shape)
#             xa = self.sigmoid(xa) if self.norm else xa
#             # .sum((2, 3)) sum func has big problem, which make pro
#             # collapse when GPU memory has half empty
#             y = xb.mul(xa)
#             y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#         return results
#
#
# class CamOvFc(NoAttention):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(CamOvFc, self).__init__(attributes, in_features, out_features, norm_size)
#         self.norm = norm_size[0]
#         # Also define attention layer for attribute
#         for attr in self.attributes:
#             name = attr.name
#             # 10 prototype just for test
#             # setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
#             # setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (3, 3), padding=1))
#             # setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 1, (1, 1)))
#             setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (1, 1)))
#             setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (1, 1)))
#             setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 1, (1, 1)))
#         self.softmax = torch.nn.Softmax(dim=2)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.global_max_pool = nn.AdaptiveMaxPool2d(1)
#         self.tanh = torch.nn.Tanh()
#
#     def forward(self, x):
#         results = []
#         cam_results = []
#         for attr in self.attributes:
#             name = attr.name
#             cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
#             cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))
#             cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
#             # cv3_rl_sigmoid = self.sigmoid(cv3_rl) if self.norm else cv3_rl
#             # cv3_rl = self.relu(cv3_rl) if self.norm else cv3_rl
#
#             # compute softmax to cam
#             cam_softmax = self.softmax(cv3_rl.view(cv3_rl.size(0), cv3_rl.size(1), -1))
#             # cam_pool = self.pool(cam_softmax.view(cv3_rl.size()))
#             cam = self.global_max_pool(cam_softmax)
#
#             # y = cv3_rl_sigmoid * x
#             y = cv3_rl + x
#             y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
#             y = getattr(self, 'fc_' + name + '_1')(y)
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             # cls = F.softmax(cls, 1)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#             # results.append(cam)
#             cam_results.append(cam)
#         results.extend(cam_results)
#         return results
# class PrTp(NoAttention):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(PrTp, self).__init__(attributes, in_features, out_features, norm_size)
#         self.norm = norm_size[0]
#         # Also define attention layer for attribute
#         for attr in self.attributes:
#             name = attr.name
#             # 10 prototype just for test
#             setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
#             setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (3, 3), padding=1))
#             setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 10, (1, 1)))
#
#             # setattr(self, 'prototype_' + name + '_coe1', nn.AdaptiveAvgPool2d(1))
#             setattr(self, 'prototype_' + name + '_coe1', nn.Linear(self.in_features, int(self.in_features / 16)))
#             setattr(self, 'prototype_' + name + '_coe2', nn.Linear(int(self.in_features / 16), 10))
#         self.tanh = torch.nn.Tanh()
#
#     def forward(self, x):
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
#             cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))
#             cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
#             cv3_rl = self.sigmoid(cv3_rl) if self.norm else cv3_rl
#             # cv3_rl = self.relu(cv3_rl) if self.norm else cv3_rl
#
#             # compute prototype coefficient
#             # prototype_coe1 = self.relu(getattr(self, 'prototype_' + name + '_coe1')(
#             # self.global_pool(x).view(x.size(0), -1)))
#             # prototype_coe2 = self.sigmoid(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))
#             prototype_coe1 = getattr(self, 'prototype_' + name + '_coe1')(self.global_pool(x).view(x.size(0), -1))
#             prototype_coe2 = self.tanh(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))
#
#             # multi prototype with attention map to produce new attention map
#             new_attention = (prototype_coe2[..., None, None] * cv3_rl).sum(1, keepdim=True)
#             y = new_attention * x
#             y = getattr(self, 'global_pool')(y).view(y.size(0), -1)
#             y = getattr(self, 'fc_' + name + '_1')(y)
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#         return results
# class Custom(NoAttention):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(Custom, self).__init__(attributes, in_features, out_features, norm_size)
#         # need to try: size [3, 5, 7]  step[1, 2, 3, 4, 5]
#         size = 5
#         step = 3
#         sigma = 1.5
#         feature_size = 14
#         add = step - ((feature_size - 1) % step) if step != 1 else 0
#         self.prototype, self.prototype_num = self.prepare_prototype(size, step, feature_size, sigma)
#         # self.custom_pool = nn.AvgPool2d(size, stride=step, padding=(size-1)//2)
#         self.custom_pool = nn.AvgPool2d(size, stride=step)
#         self.attention_pad = nn.ReplicationPad2d(((size-1)//2, (size-1)//2 + add, (size-1)//2, (size-1)//2 + add))
#
#         self.norm = norm_size[0]
#         # Also define attention layer for attribute
#         for attr in self.attributes:
#             name = attr.name
#
#             setattr(self, 'attention_' + name + '_cv1', nn.Conv2d(self.in_features, 512, (3, 3), padding=1))
#             setattr(self, 'attention_' + name + '_cv2', nn.Conv2d(512, 512, (3, 3), padding=1))
#             setattr(self, 'attention_' + name + '_cv3', nn.Conv2d(512, 1, (1, 1)))
#
#             # setattr(self, 'prototype_' + name + '_coe1', nn.Linear(self.in_features, int(self.in_features / 16)))
#             # setattr(self, 'prototype_' + name + '_coe2', nn.Linear(int(self.in_features / 16), self.prototype_num))
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(0.3)
#
#     def prepare_prototype(self, size, step, feature_size, sigma):
#
#         col = torch.Tensor([np.arange(size)] * size).view((size, size))
#         row = torch.transpose(torch.Tensor([np.arange(size)] * size).view((size, size)), 0, 1)
#         attention_ceil = torch.exp(-1 * (((col - (size-1)/2) ** 2) / sigma ** 2 + ((row - (size-1)/2) ** 2) / sigma ** 2))
#
#         # feature_size = 14
#         length = int(math.ceil((feature_size - 1) / step))
#         feature_size_supple = length * step + 1
#         prototype = torch.zeros((1, (length + 1) ** 2, feature_size_supple, feature_size_supple))
#         margin1 = int(math.ceil((size - 1) / 2))
#         margin2 = size - 1 - margin1
#         channal_num = -1
#         # row
#         for i in range(0, feature_size_supple, step):
#             # print('i', i)
#             # col
#             for j in range(0, feature_size_supple, step):
#                 # print('j', j)
#                 channal_num += 1
#                 row1 = i - margin1
#                 row2 = i + margin2
#                 col1 = j - margin1
#                 col2 = j + margin2
#
#                 row11 = 0 if row1 >= 0 else -row1
#                 row12 = size - 1 if row2 < feature_size_supple else (size-1)-(row2-feature_size_supple+1)
#                 col11 = 0 if col1 >= 0 else -col1
#                 col12 = size - 1 if col2 < feature_size_supple else (size-1)-(col2-feature_size_supple+1)
#                 row1 = row1 if row1 >= 0 else 0
#                 col1 = col1 if col1 >= 0 else 0
#                 row2 = row2 if row2 < feature_size_supple else feature_size_supple
#                 col2 = col2 if col2 < feature_size_supple else feature_size_supple
#
#                 prototype[0, channal_num, row1:row2+1, col1:col2+1] = attention_ceil[row11:row12+1, col11:col12+1]
#
#         prototype = prototype[:, :, :feature_size, :feature_size]
#         return prototype.cuda(), (length + 1) ** 2
#
#     def forward(self, x):
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             cv1_rl = self.relu(getattr(self, 'attention_' + name + '_cv1')(x))
#             cv2_rl = self.relu(getattr(self, 'attention_' + name + '_cv2')(cv1_rl))
#             cv3_rl = self.relu(getattr(self, 'attention_' + name + '_cv3')(cv2_rl))
#             cv3_rl = self.sigmoid(self.custom_pool(self.attention_pad(cv3_rl))).view(x.size(0), -1)
#             # cv3_rl = self.sigmoid(cv3_rl) if self.norm else cv3_rl
#             # cv3_rl = self.relu(cv3_rl) if self.norm else cv3_rl
#
#             # compute prototype coefficient
#             # prototype_coe1 = self.relu(getattr(self, 'prototype_' + name + '_coe1')(
#             # self.global_pool(x).view(x.size(0), -1)))
#             # prototype_coe2 = self.sigmoid(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))
#             # prototype_coe1 = getattr(self, 'prototype_' + name + '_coe1')(self.global_pool(x).view(x.size(0), -1))
#             # prototype_coe2 = self.tanh(getattr(self, 'prototype_' + name + '_coe2')(prototype_coe1))
#
#             # multi prototype with attention map to produce new attention map
#             # new_attention = (prototype_coe2[..., None, None] * self.prototype).sum(1, keepdim=True)
#             new_attention = (cv3_rl[..., None, None] * self.prototype).sum(1, keepdim=True)
#             y = new_attention * x
#             y = self.dropout(getattr(self, 'global_pool')(y).view(y.size(0), -1))
#             y = self.dropout(getattr(self, 'fc_' + name + '_1')(y))
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#
#         return results
#
#
# class TwoLevel(Base):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(TwoLevel, self).__init__(attributes, in_features, out_features, norm_size[1])
#
#         # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
#
#         # step or margin, one of them have to be 1
#         # self.length = 1
#         # self.step = 1   # distance between two groups
#         #
#         # self.in_features = (512 - (self.length - 1)) // self.step
#         # self.in_features = self.in_features + 1 if (512 - (self.length - 1)) % self.step else self.in_features
#
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         for attr in self.attributes:
#             setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x1, x2, x3 = x
#         x2 = 0.5 * self.dropout(self.global_pool(x2).view(x2.size(0), -1))
#         x3 = self.dropout(self.global_pool(x3).view(x3.size(0), -1))
#         x = torch.cat((x2, x3), 1)
#         # temp = 0
#         # end = -(self.length - 1)
#         # for l in range(self.length):
#         #     # end = -(self.length * self.margin) + 1 + l*self.margin
#         #     if end >= 0:
#         #         temp = temp + x[:, l::self.step]
#         #     else:
#         #         temp = temp + x[:, l:end:self.step]
#         #     end += 1
#         # x = temp
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             y = self.dropout(getattr(self, 'fc_' + name + '_1')(x))
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#         return results
#
#
# class TwoLevelAuto(Base):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(TwoLevelAuto, self).__init__(attributes, in_features, out_features, norm_size[1])
#
#         # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
#
#         # step or margin, one of them have to be 1
#         # self.length = 1
#         # self.step = 1   # distance between two groups
#         #
#         # self.in_features = (512 - (self.length - 1)) // self.step
#         # self.in_features = self.in_features + 1 if (512 - (self.length - 1)) % self.step else self.in_features
#
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         for attr in self.attributes:
#             setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))
#         self.dropout = nn.Dropout(0.1)
#         self.scale = Scale(0.3)
#
#     def forward(self, x):
#         x1, x2, x3 = x
#         x2 = self.scale(self.dropout(self.global_pool(x2).view(x2.size(0), -1)))
#         x3 = self.dropout(self.global_pool(x3).view(x3.size(0), -1))
#         x = torch.cat((x2, x3), 1)
#
#         # temp = 0
#         # end = -(self.length - 1)
#         # for l in range(self.length):
#         #     # end = -(self.length * self.margin) + 1 + l*self.margin
#         #     if end >= 0:
#         #         temp = temp + x[:, l::self.step]
#         #     else:
#         #         temp = temp + x[:, l:end:self.step]
#         #     end += 1
#         # x = temp
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             y = self.dropout(getattr(self, 'fc_' + name + '_1')(x))
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#         return results
#
#
# class ThreeLevel(Base):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(ThreeLevel, self).__init__(attributes, in_features, out_features, norm_size[1])
#
#         # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
#
#         # step or margin, one of them have to be 1
#         # self.length = 1
#         # self.step = 1   # distance between two groups
#         #
#         # self.in_features = (512 - (self.length - 1)) // self.step
#         # self.in_features = self.in_features + 1 if (512 - (self.length - 1)) % self.step else self.in_features
#
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         for attr in self.attributes:
#             setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         x1, x2, x3 = x
#         x1 = self.dropout(self.global_pool(x1).view(x1.size(0), -1))
#         x2 = self.dropout(self.global_pool(x2).view(x2.size(0), -1))
#         x3 = self.dropout(self.global_pool(x3).view(x3.size(0), -1))
#         x = torch.cat((x1, x2, x3), 1)
#         # temp = 0
#         # end = -(self.length - 1)
#         # for l in range(self.length):
#         #     # end = -(self.length * self.margin) + 1 + l*self.margin
#         #     if end >= 0:
#         #         temp = temp + x[:, l::self.step]
#         #     else:
#         #         temp = temp + x[:, l:end:self.step]
#         #     end += 1
#         # x = temp
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             y = self.dropout(getattr(self, 'fc_' + name + '_1')(x))
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#         return results
#
#
# class TwoLevelAlone(Base):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(TwoLevelAlone, self).__init__(attributes, in_features, out_features, norm_size[1])
#
#         # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
#
#         # step or margin, one of them have to be 1
#         # self.length = 1
#         # self.step = 1   # distance between two groups
#         #
#         # self.in_features = (512 - (self.length - 1)) // self.step
#         # self.in_features = self.in_features + 1 if (512 - (self.length - 1)) % self.step else self.in_features
#
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         for attr in self.attributes:
#             setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         x1, x2, x3 = x
#         # x2 = 0. * self.dropout(self.global_pool(x2).view(x2.size(0), -1))
#         # x3 = self.dropout(self.global_pool(x3).view(x3.size(0), -1))
#         # x = torch.cat((x2, x3), 1)
#
#         x2 = self.dropout(self.global_pool(x2).view(x2.size(0), -1))
#         x3 = self.dropout(self.global_pool(x3).view(x3.size(0), -1))
#         # x = torch.cat((x2, x3), 1)
#
#         # temp = 0
#         # end = -(self.length - 1)
#         # for l in range(self.length):
#         #     # end = -(self.length * self.margin) + 1 + l*self.margin
#         #     if end >= 0:
#         #         temp = temp + x[:, l::self.step]
#         #     else:
#         #         temp = temp + x[:, l:end:self.step]
#         #     end += 1
#         # x = temp
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             y1 = self.dropout(getattr(self, 'fc_' + name + '_1')(x3))
#             y1 = self.relu(y1)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y1)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y1)
#                 results.append(recognizable)
#
#         for attr in self.attributes:
#             name = attr.name
#             y2 = self.dropout(getattr(self, 'fc_' + name + '_1')(x2))
#             y2 = self.relu(y2)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y2)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y2)
#                 results.append(recognizable)
#         return results
#
#
# class ThreeLevelRNN(Base):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(ThreeLevelRNN, self).__init__(attributes, in_features, out_features, norm_size[1])
#
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         for attr in self.attributes:
#             setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))
#         self.dropout = nn.Dropout(0.3)
#         self.rcnn = nn.LSTM(self.in_features, self.in_features, 1)
#         self.linear_x1 = nn.Linear(256, self.in_features)
#
#     def forward(self, x):
#         x1, x2, x3 = x
#         x1 = self.linear_x1(self.global_pool(x1).view(x1.size(0), -1))
#         x2 = self.global_pool(x2).view(x2.size(0), -1)
#         x3 = self.global_pool(x3).view(x3.size(0), -1)
#
#         batch_size = x1.size(0)
#         h0 = torch.zeros(1, batch_size, self.in_features).cuda()
#         c0 = torch.zeros(1, batch_size, self.in_features).cuda()
#         x = torch.stack((x1, x2, x3), dim=0)  # (3, 60, 512)
#         output, (hn, cn) = self.rcnn(x, (h0, c0))
#         x = hn[1, :, :]
#         # temp = 0
#         # end = -(self.length - 1)
#         # for l in range(self.length):
#         #     # end = -(self.length * self.margin) + 1 + l*self.margin
#         #     if end >= 0:
#         #         temp = temp + x[:, l::self.step]
#         #     else:
#         #         temp = temp + x[:, l:end:self.step]
#         #     end += 1
#         # x = temp
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             y = self.dropout(getattr(self, 'fc_' + name + '_1')(x))
#             y = self.relu(y)
#             cls = getattr(self, 'fc_' + name + '_classifier')(y)
#             results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#         return results
#
#
# class BaseWithMulti(nn.Module):
#     def __init__(self, attributes, in_features, out_features, img_size):
#         for attr in attributes:
#             assert isinstance(attr, Attribute)
#         super(BaseWithMulti, self).__init__()
#         self.img_size = img_size
#         # self.map_size = int(self.img_size / 224 * 7)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.attributes = attributes
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#
#         for attr in self.attributes:
#             name = attr.name
#             if attr.branch_num == 1:
#                 setattr(self, 'fc_' + name + '_classifier', nn.Linear(512, 1))
#             else:
#                 for i in range(attr.branch_num):
#                     setattr(self, 'fc_' + name + '_classifier' + str(i), nn.Linear(512, 1))
#             # Also define a branch for classifying recognizability if necessary
#             if attr.rec_trainable:
#                 setattr(self, 'fc_' + name + '_recognizable', nn.Linear(512, 1))
#             # setattr(self, 'fc_' + name + '_1', nn.Linear(self.in_features, 512))
#
#     def forward(self, *parameters):
#         r"""Defines the computation performed at every call.
#
#         Should be overridden by all subclasses.
#
#         .. note::
#             Although the recipe for forward pass needs to be defined within
#             this function, one should call the :class:`Module` instance afterwards
#             instead of this since the former takes care of running the
#             registered hooks while the latter silently ignores them.
#         """
#         raise NotImplementedError
#
#
# class NoAttentionMuti(BaseWithMulti):
#     def __init__(self, attributes, in_features, out_features, norm_size):
#         super(NoAttentionMuti, self).__init__(attributes, in_features, out_features, norm_size[1])
#
#         # self.global_pool = nn.AvgPool2d((self.map_size, self.map_size), stride=1)
#
#         # step or margin, one of them have to be 1
#
#         # self.length = 7
#         # self.step = 5   # distance between two groups
#         #
#         # self.in_features = (512 - (self.length - 1)) // self.step
#         # self.in_features = self.in_features + 1 if (512 - (self.length - 1)) % self.step else self.in_features
#
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         for attr in self.attributes:
#             setattr(self, 'fc_' + attr.name + '_1', nn.Linear(self.in_features, 512))
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         x = self.dropout(self.global_pool(x).view(x.size(0), -1))
#         # temp = 0
#         # end = -(self.length - 1)
#         # for l in range(self.length):
#         #     # end = -(self.length * self.margin) + 1 + l*self.margin
#         #     if end >= 0:
#         #         temp = temp + x[:, l::self.step]
#         #     else:
#         #         temp = temp + x[:, l:end:self.step]
#         #     end += 1
#         # x = temp
#         results = []
#         for attr in self.attributes:
#             name = attr.name
#             y = self.dropout(getattr(self, 'fc_' + name + '_1')(x))
#             y = self.relu(y)
#             if attr.branch_num == 1:
#                 cls = getattr(self, 'fc_' + name + '_classifier')(y)
#                 results.append(cls)
#             else:
#                 for i in range(attr.branch_num):
#                     cls = getattr(self, 'fc_' + name + '_classifier' + str(i))(y)
#                     results.append(cls)
#             if attr.rec_trainable:  # Also return the recognizable branch if necessary
#                 recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
#                 results.append(recognizable)
#         return results