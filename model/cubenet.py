import torch.nn as nn
from model.model_utils import get_backbone_network, get_param_groups
from model import model_utils
from data.attributes import Attribute
import torch


class CubeNet(nn.Module):
    def __init__(self, conv, attributes, pretrained=True, img_size=224, attention=None, norm=False):
        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)
        assert isinstance(attributes, list)
        for attr in attributes:
            assert isinstance(attr, Attribute)

        super(CubeNet, self).__init__()

        self.imag_size = img_size
        self.attention = attention
        self.norm = norm
        self.conv = conv
        self.feature_extractor, self.feature_map_depth, self.mean, self.std = get_backbone_network(conv, pretrained)

        # TODO Test if using separate spatial attention pooling instead of global pooling for each branch works better
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attributes = attributes
        self.classifier = getattr(model_utils, self.attention)(
            self.attributes, self.feature_map_depth, 2, (self.norm, self.imag_size))

    def forward(self, x):
        # Need to override forward method as JIT/trace requires call to _slow_forward, which is called implicitly by
        # __call__ only. Simply calling forward() or features() will result in missing scope name in traced graph.
        # Override here so that in multi-gpu training case when model is replicated, the forward function of feature
        # extractor still gets overriden
        self.feature_extractor.forward = self.feature_extractor.features
        x = self.feature_extractor(x)
        result = self.classifier(x)
        return result

# TODO use it to do something, now has no use for all project
    def get_parameter_groups(self):
        """
        Return model parameters in three groups: 1) First Conv layer of the CNN feature extractor; 2) Other Conv layers
        of the CNN feature extractor; 3) All other added layers of this model
        """
        if self.conv.startswith('resnet'):
            first_conv = [self.feature_extractor.conv1, self.feature_extractor.bn1]  # Only support Resnet for now
            other_convs = [self.feature_extractor.layer1, self.feature_extractor.layer2, self.feature_extractor.layer3,
                           self.feature_extractor.layer4]
            new_layers = [module for module_name, module in self.named_modules() if module_name.startswith('fc')]
            param_groups = [first_conv, other_convs, new_layers]
        else:
            first_conv = [self.feature_extractor._features]  # Only support VGG for now
            new_layers = [module for module_name, module in self.named_modules() if module_name.startswith('fc') or
                          module_name.startswith('attention_')]
            param_groups = [first_conv, new_layers]
        return get_param_groups(param_groups)
