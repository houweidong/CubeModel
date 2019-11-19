import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from data.attributes import AttributeType as AT, Attribute

Compose = transforms.Compose


def get_inference_transform(mean, std, face_size=224):
    def inference_transform(input):
        img, bbox = input
        w, h = img.size

        xc = (bbox[0] + bbox[2]) / 2
        yc = (bbox[1] + bbox[3]) / 2
        wbox = (bbox[2] - bbox[0]) / 2
        hbox = (bbox[3] - bbox[1]) / 2

        # Crop a square patch with added margin
        box_size = min(w - xc, h - yc, xc, yc, wbox * 1.4, hbox * 1.4)
        crop = img.crop((xc - box_size, yc - box_size, xc + box_size, yc + box_size))

        # Convert to normalized Pytorch Tensor
        return F.normalize(F.to_tensor(F.resize(crop, (face_size, face_size))), mean, std)

    return inference_transform


def inference_transform(input):
    img, bbox = input
    w, h = img.size

    xc = (bbox[0] + bbox[2]) / 2
    yc = (bbox[1] + bbox[3]) / 2
    wbox = (bbox[2] - bbox[0]) / 2
    hbox = (bbox[3] - bbox[1]) / 2

    # Crop a square patch with added margin
    box_size = min(w - xc, h - yc, xc, yc, wbox * 1.4, hbox * 1.4)
    crop = img.crop((xc - box_size, yc - box_size, xc + box_size, yc + box_size))

    return crop


def get_inference_transform_person(input):
    img, bbox = input

    # bbox = [xmin, ymin, w, h]
    return img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

def get_inference_transform_person_lr(input):
    img, bbox = input

    # bbox = [xmin, ymin, xmax, ymax]
    result = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    # if result:
    #     print(bbox)
    #     return result
    # else:
    #     print('NOne')
    #     print(bbox)
    return result

def square_no_elastic(img):
    w, h = img.size
    size = max(w, h)
    return F.center_crop(img, size)


# for branch mode
# Convert target of a sample to Tensor(s). Support both multi-dataset setting as well as recognizability.
# Assuming each sample is a dict whose keys include Attributes as well as recognizability of each attribute
class ToMaskedTargetTensor(object):
    _tensor_types = {AT.NUMERICAL: torch.float, AT.BINARY: torch.long, AT.MULTICLASS: torch.long}

    def __init__(self, attributes):
        for attr in attributes:
            assert isinstance(attr, Attribute)

        self.attrs = attributes

    def __call__(self, sample):
        mask = []  # Mask indicating whether each attribute is available in this sample
        target = []
        dummy_val = -10  # Placeholder value for unavailable attribute, so that each sample has the same length
        for i, attr in enumerate(self.attrs):
            # if attr.key in sample:
            recognizability = sample['recognizability'][attr.key]
            if attr.branch_num == 1:
                # Class label is valid(available) only when the attribute of this sample is recognizable
                if recognizability == 1:
                    cls_available = 1
                    val = sample[attr.key]
                else:
                    cls_available = 0
                    val = dummy_val
                rec_available = 1  # Recognizability is available as long as the sample contains the attribute
                # Use a mask tensor to indicate which attribute is available on each sample
                mask.append(torch.tensor([cls_available], dtype=torch.uint8, requires_grad=False))
                target.append(torch.tensor([val], dtype=self._tensor_types[attr.data_type], requires_grad=False))
            else:
                for j in range(attr.branch_num):
                    # Class label is valid(available) only when the attribute of this sample is recognizable
                    cls_available = 1
                    if recognizability == 1:
                        if j == sample[attr.key]:
                            val = 1
                        else:
                            val = 0
                    else:
                        cls_available = 0
                        val = dummy_val
                    rec_available = 1  # Recognizability is available as long as the sample contains the attribute
                    # Use a mask tensor to indicate which attribute is available on each sample
                    mask.append(torch.tensor([cls_available], dtype=torch.uint8, requires_grad=False))
                    target.append(torch.tensor([val], dtype=self._tensor_types[attr.data_type], requires_grad=False))
                    # if val == 2:
                    #     a = 5

            # else:
            #     cls_available = 0
            #     val = dummy_val
            #     rec_available = 0
            #     if attr.rec_trainable:
            #         recognizability = dummy_val

            if attr.rec_trainable:
                # When one attribute's recognizability is trainable, we always return a tuple,
                # no matter this sample contains such info or not
                mask.append(torch.tensor([rec_available], dtype=torch.uint8, requires_grad=False))
                target.append(torch.tensor([recognizability], dtype=torch.long, requires_grad=False))
        # print(target)
        return target, mask


# for paper mode in the test period
# there is no recognizability for each attribute
class ToMaskedTargetTensorPaper(object):
    _tensor_types = {AT.NUMERICAL: torch.float, AT.BINARY: torch.long, AT.MULTICLASS: torch.long}

    def __init__(self, attributes):
        for attr in attributes:
            assert isinstance(attr, Attribute)

        self.attrs = attributes

    def __call__(self, sample):
        mask = []  # Mask indicating whether each attribute is available in this sample
        target = []
        dummy_val = -10  # Placeholder value for unavailable attribute, so that each sample has the same length
        for i, attr in enumerate(self.attrs):
            if attr.key in sample:
                recognizability = sample['recognizability'][attr.key]
                # Class label is valid(available) only when the attribute of this sample is recognizable
                if recognizability == 1:
                    cls_available = 1
                    val = sample[attr.key]
                else:
                    cls_available = 0
                    val = dummy_val
            else:
                cls_available = 0
                val = dummy_val

            # Use a mask tensor to indicate which attribute is available on each sample
            mask.append(torch.tensor([cls_available], dtype=torch.uint8, requires_grad=False))
            target.append(torch.tensor([val], dtype=self._tensor_types[attr.data_type], requires_grad=False))

        return target, mask
