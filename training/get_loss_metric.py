from ignite.metrics import CategoricalAccuracy, Loss, Accuracy
from ignite.contrib.metrics import AveragePrecision
from training.loss_utils import get_categorial_loss, reverse_ohem_loss, exp_loss
from data.attributes import AttributeType
import torch.nn.functional as F


def get_losses_metrics(attrs, categorical_loss='cross_entropy', attention='None'):
    loss_fn = get_categorial_loss(categorical_loss)
    losses, metrics = [], []
    cam_losses = []

    for attr in attrs:
        # For attribute classification
        losses.append(loss_fn)
        if attr.data_type == AttributeType.BINARY:
            metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(loss_fn)])
        elif attr.data_type == AttributeType.MULTICLASS:
            metrics.append([Accuracy(), Loss(loss_fn)])
        elif attr.data_type == AttributeType.NUMERICAL:
            # not support now
            pass
        # For recognizability classification
        if attr.rec_trainable:
            metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(reverse_ohem_loss)])
            # Always use reverse OHEM loss for recognizability, at least for now
            losses.append(reverse_ohem_loss)
        if attention == 'CamOvFc':
            # losses.append(exp_loss)
            cam_losses.append(exp_loss)
    losses.extend(cam_losses)

    return losses, metrics
