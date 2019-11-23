import torch
import torch.nn.functional as F
import math
from functools import partial
from data.attributes import NewAttributes


def exp_loss(pred, alpha=-23, beta=-18):

    loss = 0.0 * torch.exp(alpha * (pred + beta / float(7 * 7)))
    return loss.mean()

# alpha now only support for binary classification
# TODO Change it to class so that gamma can also be learned
def focal_loss(pred, target_float, gamma=2, alpha=None, size_average=True):
    target = torch.round(target_float).long()
    if isinstance(alpha, (float, int)):
        alpha = torch.Tensor([alpha, 1 - alpha])
    if isinstance(alpha, list):
        alpha = torch.Tensor(alpha)

    target = target.view(-1, 1)
    target_float = target_float.view(-1, 1)
    # logpt = F.log_softmax(pred, 1)
    # logpt = logpt.gather(1, target)
    # pt = logpt.exp()
    # ls = F.logsigmoid(pred)
    # ls_1m = 1 - ls
    pt = torch.sigmoid(pred)
    pt_1m = 1 - pt

    logpt = torch.log(torch.cat((pt_1m, pt), dim=1)).gather(1, target)
    logpt = logpt.view(-1)

    pt_final = torch.cat((pt_1m, pt), dim=1).gather(1, target).view(-1)

    if alpha is not None:
        if alpha.type() != pred.data.type():
            alpha = alpha.type_as(pred.data)
        at = alpha.gather(0, target.data.view(-1))
        logpt = logpt * at
    loss1_coe = torch.cat((1-target_float, target_float), dim=1).gather(1, target)
    loss1 = (-1 * (1 - pt_final) ** gamma * logpt) * loss1_coe

    logpt1 = torch.log(torch.cat((pt, pt_1m), dim=1)).gather(1, target)
    logpt1 = logpt1.view(-1)

    pt_final = torch.cat((pt, pt_1m), dim=1).gather(1, target).view(-1)

    if alpha is not None:
        if alpha.type() != pred.data.type():
            alpha = alpha.type_as(pred.data)
        at = alpha.gather(0, target.data.view(-1))
        logpt1 = logpt1 * at
    loss2 = (-1 * (1 - pt_final) ** gamma * logpt1) * (1 - loss1_coe)
    # loss2_coe = 1 - loss1_coe
    loss = loss1 + loss2
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def binary_cn(pred, target, weight=None):
    return F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float(), weight=weight)


# to solve the imbalance problem
def ohem_loss(pred, target_float, ratio=3, reverse=False):
    assert pred.size()[1] == 2 or pred.size()[1] == 1  # Only support binary case
    target = torch.round(target_float)
    # print(target)
    if not reverse:
        pos_mask = target.byte()
        neg_mask = 1 - pos_mask
    else:  # Simply reversing mask for positive/negative samples will yield reverse OHEM
        neg_mask = target.byte()
        pos_mask = 1 - neg_mask

    n_pos = int(torch.sum(pos_mask))
    n_neg = int(torch.sum(neg_mask))
    if n_neg > 0 and n_neg > n_pos * ratio:

        n_selected = max(n_pos * ratio, 1)

        ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target_float, reduction='none')
        # ce_loss = F.cross_entropy(pred, target, reduction='none')

        # generate top k neg ce loss mask
        loss_neg_samples = torch.masked_select(ce_loss, neg_mask)
        _, index = torch.topk(loss_neg_samples, n_selected)

        # Get mask of selected negative samples on original mask tensor
        selected_neg_mask = torch.zeros(int(n_neg), device='cuda')
        selected_neg_mask.scatter_(0, index, 1)  # a [n_neg] size mask
        # print(n_pos, n_neg, neg_mask.size())
        neg_index = torch.masked_select(torch.arange(n_pos + n_neg, dtype=torch.long, device='cuda', requires_grad=False),
                                        neg_mask)  # Mapping from [n_neg] to [n_pos+n_neg] mask
        neg_mask.scatter_(0, neg_index, selected_neg_mask.byte())
        # Return average loss of all selected samples
        mask = neg_mask + pos_mask
        masked_loss = torch.masked_select(ce_loss, mask)

        # anp = torch.masked_select(pred[:, 0], neg_mask).mean()
        # app = torch.masked_select(pred[:, 1], pos_mask).mean()
        # np_contrast = anp / app
        return masked_loss.mean()  # , np_contrast
    else:
        # anp = torch.masked_select(pred[:, 0], neg_mask).mean()
        # app = torch.masked_select(pred[:, 1], pos_mask).mean()
        # np_contrast = anp / app
        # return F.cross_entropy(pred, target)  # , np_contrast
        return F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float())


class Ohem(object):

    def __init__(self, state=True, pos_length=1000, neg_length=1000):
        self.state = state
        self.pos_length = int(pos_length)
        self.neg_length = int(neg_length)
        self.pos_pool = []
        self.neg_pool = []
        self.distance = 0
        self.ratio = 1

        # super param
        self.mi = 2
        self.a = 3
        self.base = math.e

    def __call__(self, pred, target):
        assert pred.size()[1] == 2 or pred.size()[1] == 1  # Only support binary case

        pos_mask = target.byte()
        neg_mask = 1 - pos_mask

        n_pos = int(torch.sum(pos_mask))
        n_neg = int(torch.sum(neg_mask))

        # if self.state:
        if n_pos > 0:
            self.pos_pool.extend(list(torch.masked_select(torch.sigmoid(pred[:, 0]), pos_mask).detach().cpu().numpy()))
        if n_neg > 0:
            self.neg_pool.extend(list(torch.masked_select(torch.sigmoid(pred[:, 0]), neg_mask).detach().cpu().numpy()))
        if len(self.pos_pool) > self.pos_length:
            self.pos_pool = self.pos_pool[-self.pos_length:]
        if len(self.neg_pool) > self.neg_length:
            self.neg_pool = self.neg_pool[-self.neg_length:]

        if len(self.pos_pool) >= self.pos_length and len(self.neg_pool) >= self.neg_length:
            pos_mean = 1 - sum(self.pos_pool) / len(self.pos_pool)
            neg_mean = sum(self.neg_pool) / len(self.neg_pool)
            self.distance = neg_mean - pos_mean
            self.ratio = math.pow(self.base, (1 - abs(self.distance)) ** self.mi * self.a)

        print("distance: ", self.distance)
        print("ratio: ", self.ratio)
        if self.state and abs(self.distance) > 0.1 and \
                len(self.pos_pool) >= self.pos_length and len(self.neg_pool) >= self.neg_length and \
                ((self.distance < 0 and n_neg > 0 and n_neg > n_pos * self.ratio) or
                 (self.distance > 0 and n_pos > 0 and n_pos > n_neg * self.ratio)):

            if self.distance < 0:
                n_selected = int(max(n_pos * self.ratio, 1))

                ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float(), reduction='none')
                # ce_loss = F.cross_entropy(pred, target, reduction='none')

                # generate top k neg ce loss mask
                loss_neg_samples = torch.masked_select(ce_loss, neg_mask)
                _, index = torch.topk(loss_neg_samples, n_selected)

                # Get mask of selected negative samples on original mask tensor
                selected_neg_mask = torch.zeros(int(n_neg), device='cuda')
                selected_neg_mask.scatter_(0, index, 1)  # a [n_neg] size mask
                neg_index = torch.masked_select(
                    torch.arange(n_pos + n_neg, dtype=torch.long, device='cuda', requires_grad=False),
                    neg_mask)  # Mapping from [n_neg] to [n_pos+n_neg] mask
                neg_mask.scatter_(0, neg_index, selected_neg_mask.byte())

                # Return average loss of all selected samples
                mask = neg_mask + pos_mask
                masked_loss = torch.masked_select(ce_loss, mask)
            else:
                n_selected = int(max(n_neg * self.ratio, 1))

                ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float(), reduction='none')
                # ce_loss = F.cross_entropy(pred, target, reduction='none')

                # generate top k pos ce loss mask
                loss_pos_samples = torch.masked_select(ce_loss, pos_mask)
                _, index = torch.topk(loss_pos_samples, n_selected)

                # Get mask of selected pos samples on original mask tensor
                selected_pos_mask = torch.zeros(int(n_pos), device='cuda')
                selected_pos_mask.scatter_(0, index, 1)  # a [n_neg] size mask
                pos_index = torch.masked_select(
                    torch.arange(n_pos + n_neg, dtype=torch.long, device='cuda', requires_grad=False),
                    pos_mask)  # Mapping from [n_neg] to [n_pos+n_neg] mask
                pos_mask.scatter_(0, pos_index, selected_pos_mask.byte())

                # Return average loss of all selected samples
                mask = neg_mask + pos_mask
                masked_loss = torch.masked_select(ce_loss, mask)

            # print(self.ratio)
            return masked_loss.mean()  # , np_contrast
        else:
            return F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float())


def reverse_ohem_loss(pred, target, ratio=3): return ohem_loss(pred, target, ratio, reverse=True)


def get_categorial_loss(attrs, loss):
    if loss == 'cross_entropy':
        loss_fns = {}
        for attr in attrs:
            loss_fns[attr] = []
            loss_fns[attr].append(binary_cn)
            if attr.rec_trainable:
                loss_fns[attr].append(binary_cn)
        return loss_fns
    elif loss == 'cross_entropy_weight':
        # return F.cross_entropy, F.cross_entropy
        loss_fns = {}
        weights = get_categorial_weight()
        for attr in attrs:
            loss_fns[attr] = []
            loss_fns[attr].append(partial(binary_cn, weight=weights[attr.key][0]))
            if attr.rec_trainable:
                loss_fns[attr].append(partial(binary_cn, weight=weights[attr.key][1]))

        return loss_fns
    elif loss == 'ohem':
        loss_fns = {}
        for attr in attrs:
            loss_fns[attr] = []
            loss_fns[attr].append(ohem_loss)
            if attr.rec_trainable:
                loss_fns[attr].append(reverse_ohem_loss)
        return loss_fns
        # return Ohem, ohem_loss
    elif loss == 'focal':
        loss_fns = {}
        weights = get_categorial_weight()
        for attr in attrs:
            loss_fns[attr] = []
            loss_fns[attr].append(partial(focal_loss, alpha=weights[attr.key][0] / (weights[attr.key][0] + 1)))
            if attr.rec_trainable:
                loss_fns[attr].append(partial(focal_loss, alpha=weights[attr.key][1] / (weights[attr.key][1] + 1)))
        return loss_fns
        # return focal_loss, focal_loss
    else:
        raise Exception("Loss '{}' is not supported".format(loss))


# def get_categorial_scale():
#
#     scales = [(10263+2032)/16436, (19092+3243)/6396, (26284+991)/1456, (21674+422)/6635, (20991+1947)/5793,
#                 (13339+1879)/13513, (26200+273)/2258, (14120+10369)/4242, (18731+7585)/2415, (8168+10010)/10553,
#                 (18275+7571)/2885, (26622+1101)/1008, (19045+1252)/8434, (26507+229)/1995]
#
#     pos_num = [16436, 6396, 1456, 6635, 5793, 13513, 2258, 4242, 2415, 10553, 2885, 1008, 8434, 1995]
#     result = []
#     for scale in scales:
#         # result.append(1/(1+scale))
#         # if 0 <= scale < 5:
#         #     result.append(0.5)
#         # elif 5 <= scale < 10:
#         #     result.append(1/3)
#         # elif 10 <= scale:
#         #     result.append(0.25)
#         result.append(scale)
#
#     return result, pos_num


def get_categorial_weight():

    weight = {}
    weight[NewAttributes.yifujinshen_yesno] = [(25187) / 6650, 997 / (6650 + 25187)]
    weight[NewAttributes.kuzijinshen_yesno] = [(13645) / 7298, 11891 / (7298 + 13645)]
    weight[NewAttributes.maozi_yesno] = [(20080) / 3515, 9239 / (3515 + 20080)]
    weight[NewAttributes.gaolingdangbozi_yesno] = [(21930) / 8509, 2395 / (8509 + 21930)]
    weight[NewAttributes.gaofaji_yesno] = [(11106) / 9046, 12682 / (11106 + 9046)]

    # just for classification for two class
    return weight


# TODO Add coefficient to losses
def multitask_loss(output, label, loss_fns):
    """
    Combine losses of each branch(attribute) of multi-task learning model and return a single total loss,
    which is ready for back-propagation. This function also handles multi-dataset training where
    each sample of the input training batch may come from different datasets and thus contains different
    subset of branches in output (specified by mask in target).
    :param output: Predictions from model w.r.t training batch.
    :param label: Groundtruth of the training batch.
    :param loss_fns: A list containing losses of each branch of output of the model, following the same order.
    :return: A Pytorch Tensor that sum losses of each branch.
    """
    target, mask = label
    # n_samples = target[0].size()[0]
    n_tasks = len(target)
    # index = torch.arange(n_samples, dtype=torch.long, device='cuda')
    total_loss = 0
    for i in range(n_tasks):
        # Only add loss regarding this attribute if it is present in any sample of this batch
        if mask[i].any():
            output_fil = torch.masked_select(output[i], mask[i]).view(-1, output[i].size()[1])
            gt = torch.masked_select(target[i], mask[i])
            total_loss += loss_fns[i](output_fil, gt)
    # n_tasks_remain = len(output) - n_tasks
    # for j in range(n_tasks_remain):
    #     # TODO deal with the mask condition
    #     total_loss += loss_fns[j + n_tasks](output[j + n_tasks])

    return total_loss

