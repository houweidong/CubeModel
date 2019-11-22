from ignite.metrics import Loss
from ignite.contrib.metrics import AveragePrecision
from training.loss_utils import get_categorial_loss, reverse_ohem_loss, exp_loss,\
    get_categorial_scale, get_categorial_weight
from data.attributes import AttributeType
# import torch.nn.functional as F
import torch
from functools import partial
from training.metric_utils import MyAccuracy


def get_losses_metrics(attrs, categorical_loss='cross_entropy', attention='None'):

    scales, pos_nums = get_categorial_scale()
    loss_fns = get_categorial_loss(attrs, categorical_loss)
    losses, metrics = [], []
    # cam_losses = []

    for attr, scale, pos_num, loss_fn in zip(attrs, scales, pos_nums, loss_fns):
        # For attribute classification
        # if categorical_loss in ['ohem', 'focal']:
        #     losses.append(partial(loss_fn, state=True, pos_length=pos_num / 10, neg_length=pos_num / 10 * scale)())
        # else:
        #     losses.append(loss_fn)

        if attr.data_type == AttributeType.BINARY:
            # metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(loss_fn)])
            metrics.append(
                [AveragePrecision(activation=lambda pred: torch.sigmoid(pred)),
                 MyAccuracy(output_transform=lambda pred: torch.sigmoid(pred)), Loss(loss_fn[attr][0])])
            losses.append(loss_fn[attr][0])
        elif attr.data_type == AttributeType.MULTICLASS:
            for i in range(attr.branch_num):
                metrics.append(
                    [AveragePrecision(activation=lambda pred: torch.sigmoid(pred)),
                     MyAccuracy(output_transform=lambda pred: torch.sigmoid(pred)), Loss(loss_fn[attr][0])])
                losses.append(loss_fn[attr][0])
        elif attr.data_type == AttributeType.NUMERICAL:
            # not support now
            pass
        # For recognizability classification
        if attr.rec_trainable:
            # metrics.append([AveragePrecision(activation=lambda pred: F.softmax(pred, 1)[:, 1]), Accuracy(), Loss(reverse_ohem_loss)])
            metrics.append(
                [AveragePrecision(activation=lambda pred: torch.sigmoid(pred)),
                 MyAccuracy(output_transform=lambda pred: torch.sigmoid(pred)), Loss(loss_fn[attr][1])])
            # Always use reverse OHEM loss for recognizability, at least for now
            losses.append(loss_fn[attr][1])
        # if attention == 'CamOvFc':
        #     # losses.append(exp_loss)
        #     cam_losses.append(exp_loss)

    # if attention in ['TwoLevelAlone', 'ThreeLevelAlone']:
    #     for attr, scale, pos_num in zip(attrs, scales, pos_nums):
    #         losses.append(loss_fn_val)
    #         # For recognizability classification
    #         if attr.rec_trainable:
    #             # Always use reverse OHEM loss for recognizability, at least for now
    #             losses.append(reverse_ohem_loss)
    #         if attention == 'CamOvFc':
    #             # losses.append(exp_loss)
    #             cam_losses.append(exp_loss)
    # losses.extend(cam_losses)

    return losses, metrics

