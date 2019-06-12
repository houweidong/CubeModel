from ignite.metrics import CategoricalAccuracy, Loss, Accuracy
from ignite.contrib.metrics import AveragePrecision
from training.loss_utils import get_categorial_loss, reverse_ohem_loss, exp_loss, get_categorial_scale
from data.attributes import AttributeType
# import torch.nn.functional as F
import torch
from functools import partial


def get_losses_metrics(attrs, categorical_loss='cross_entropy', attention='None', pool_num=100):
    loss_fn, loss_fn_val = get_categorial_loss(categorical_loss)
    scales = get_categorial_scale(categorical_loss)
    losses, metrics = [], []
    cam_losses = []

    for attr, scale in zip(attrs, scales):
        # For attribute classification
        if categorical_loss in ['ohem', 'focal']:
            losses.append(partial(loss_fn, pos_length=pool_num, neg_length=pool_num * scale)())
        if attr.data_type == AttributeType.BINARY:
            # metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(loss_fn)])
            metrics.append(
                [AveragePrecision(activation=lambda pred: torch.sigmoid(pred)),
                 Accuracy(output_transform=lambda pred, target: torch.sigmoid(pred)), Loss(loss_fn_val)])
        elif attr.data_type == AttributeType.MULTICLASS:
            metrics.append([Accuracy(), Loss(loss_fn_val)])
        elif attr.data_type == AttributeType.NUMERICAL:
            # not support now
            pass
        # For recognizability classification
        if attr.rec_trainable:
            # metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(reverse_ohem_loss)])
            metrics.append([AveragePrecision(activation=lambda pred: torch.sigmoid(pred)), Accuracy(),
                            Loss(reverse_ohem_loss)])
            # Always use reverse OHEM loss for recognizability, at least for now
            losses.append(reverse_ohem_loss)
        if attention == 'CamOvFc':
            # losses.append(exp_loss)
            cam_losses.append(exp_loss)
    losses.extend(cam_losses)

    return losses, metrics
