import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['vgg16', 'vgg16_bn']


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


class MySequential(nn.Sequential):

    def forward(self, input):
        result = []
        for i, module in enumerate(self._modules.values()):
            input = module(input)
            if i in [8, 12,  16]:
                result.append(input)
        return result


class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
#
#
# class VGGTwoLevel(nn.Module):
#
#     def __init__(self, features1, features2, init_weights=True):
#         super(VGGTwoLevel, self).__init__()
#         self.features1 = features1
#         self.features2 = features2
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x1 = self.features1(x)
#         x2 = self.features2(x1)
#         return x1, x2
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#
# class VGGThreeLevel(nn.Module):
#
#     def __init__(self, features1, features2, features3, init_weights=True):
#         super(VGGThreeLevel, self).__init__()
#         self.features1 = features1
#         self.features2 = features2
#         self.features3 = features3
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x1 = self.features1(x)
#         x2 = self.features2(x1)
#         x3 = self.features3(x2)
#         return x1, x2, x3
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_level(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return MySequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    #
    # 'D21': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    # 'D22': ['M', 512, 512, 512],
    #
    # 'D31': [64, 64, 'M', 128, 128, 'M', 256, 256, 256],
    # 'D32': ['M', 512, 512, 512],
    # 'D33': ['M', 512, 512, 512]
}


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_level(pretrained=False, **kwargs):

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers_level(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


# def vgg16_3level(pretrained=False, **kwargs):
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers_level(cfg['D']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
#     return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model
