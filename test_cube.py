import os
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, \
    RandomHorizontalFlip, Normalize, RandomRotation, ColorJitter

from data.transforms import square_no_elastic, get_inference_transform_person_lr
from utils.opts import parse_opts
from data.image_loader import opencv_loader, cv_to_pil_image
import cv2
from model.cubenet import CubeNet
from utils.get_tasks import get_tasks
import matplotlib.pyplot as plt

opt = parse_opts()
# opt.pretrain = False


def get_input(cuda=True, transform=None, box=None, path=None):
    pic_path = opt.img_path if not path else path
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    val_img_transform = Compose(
        [square_no_elastic,
         Resize((opt.person_size, opt.person_size)),
         ToTensor(), Normalize(mean, std)])
    img_ori = cv2.imread(pic_path)

    img = opencv_loader(pic_path)
    if transform:
        img = transform((img, box))
    img = val_img_transform(img)
    # print(img)
    img = img.unsqueeze(0)
    if cuda:
        img = img.cuda()
    return img_ori, img


def get_model(cuda=True):
    attr, _ = get_tasks(opt)
    model = CubeNet(opt.train, opt.conv, attr, pretrained=False, img_size=opt.person_size,
                    attention=opt.attention, dropout=opt.dropout, at=opt.at, at_loss=opt.at_loss)

    # load the model, need to move the prefix "module."
    state_dict = torch.load(opt.model_path, map_location='cpu')["state_dict"]
    for k in list(state_dict.keys()):
        k_new = k[7:]
        state_dict[k_new] = state_dict[k]
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)

    if cuda:
        model = model.cuda()
    model.eval()
    return model


def display(im, tensor_p):
    probs = []
    for p in tensor_p:
        probs.append(p.cpu().detach().numpy()[0, 0])
    for i, attr in enumerate(opt.specified_attrs):
        caption = "{}:{:.2f}".format(attr,  probs[i])
        im = cv2.putText(
            im, caption, (0, 20 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    cv2.imshow("result", im)
    cv2.waitKey(0)


def test_dir(model, subset='val'):
    root = "/root/dataset/new"
    # path = "/root/dataset/new/pictures/jinshenyi/"
    # label = "/root/dataset/new/labels1/jinshenyi/"

    anno_dir = os.path.join(root, 'labels1')
    anno_list = list(filter(lambda x: x.endswith('txt'), os.listdir(anno_dir)))
    for anno_txt in anno_list:
        anno_path = os.path.join(anno_dir, anno_txt)
        with open(anno_path) as f:
            lines = f.readlines()
            if subset == 'train':
                lines = [lines[i] for i in range(len(lines)) if not str(i).endswith(('3' '6', '9'))]
            else:
                lines = [lines[i] for i in range(len(lines)) if str(i).endswith(('3' '6', '9'))]
            for line in lines:
                line_list = line.split()
                if line_list:  # may have []
                    img_name = line_list[0]
                    img_path = os.path.join(root, 'pictures', anno_txt.rstrip('.txt'), img_name)
                    for i in range(1, len(line_list), 16):
                        label = line_list[i:i + 12]
                        box = list(map(lambda x: float(x), line_list[i + 12:i + 16]))
                        # there have 9 pictures' boxes have problems, so need to filter them
                        if box[2] < box[0] or box[3] < box[1]:
                            continue

                        img_ori, img = get_input(transform=get_inference_transform_person_lr, box=box, path=img_path)
                        output = model(img)
                        display(img_ori, output)


model = get_model()
if opt.test_mode == 'train_dir':
    test_dir(model)
elif opt.test_mode == 'pic':
    img_ori, img = get_input()
    output = model(img)
    display(img_ori, output)
else:
    # TODO
    pass

