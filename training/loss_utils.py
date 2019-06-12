import torch
import torch.nn.functional as F
import math

def exp_loss(pred, alpha=-23, beta=-18):

    loss = 0.0 * torch.exp(alpha * (pred + beta / float(7 * 7)))
    return loss.mean()


# alpha now only support for binary classification
# TODO Change it to class so that gamma can also be learned
def focal_loss(pred, target, gamma=2, alpha=None, size_average=True):
    if isinstance(alpha, (float, int)):
        alpha = torch.Tensor([alpha, 1 - alpha])
    if isinstance(alpha, list):
        alpha = torch.Tensor(alpha)

    target = target.view(-1, 1)
    # logpt = F.log_softmax(pred, 1)
    # logpt = logpt.gather(1, target)
    # pt = logpt.exp()
    # ls = F.logsigmoid(pred)
    # ls_1m = 1 - ls
    pt = torch.sigmoid(pred)
    pt_1m = 1 - pt

    logpt = torch.log(torch.cat((pt_1m, pt), dim=1)).gather(1, target)
    logpt = logpt.view(-1)

    pt = torch.cat((pt_1m, pt), dim=1).gather(1, target).view(-1)

    if alpha is not None:
        if alpha.type() != pred.data.type():
            alpha = alpha.type_as(pred.data)
        at = alpha.gather(0, target.data.view(-1))
        logpt = logpt * at

    loss = -1 * (1 - pt) ** gamma * logpt
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


# to solve the imbalance problem
def ohem_loss(pred, target, ratio=3, reverse=False):
    assert pred.size()[1] == 2 or pred.size()[1] == 1  # Only support binary case

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

        ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float(), reduction='none')
        # ce_loss = F.cross_entropy(pred, target, reduction='none')

        # generate top k neg ce loss mask
        loss_neg_samples = torch.masked_select(ce_loss, neg_mask)
        _, index = torch.topk(loss_neg_samples, n_selected)

        # Get mask of selected negative samples on original mask tensor
        selected_neg_mask = torch.zeros(int(n_neg), device='cuda')
        selected_neg_mask.scatter_(0, index, 1)  # a [n_neg] size mask
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

    def __init__(self, pos_length=100, neg_length=100):
        self.pos_length = pos_length
        self.neg_length = neg_length
        self.pos_pool = []
        self.neg_pool = []
        self.distance = 0
        self.ratio = 1

    def __call__(self, pred, target):
        assert pred.size()[1] == 2 or pred.size()[1] == 1  # Only support binary case

        pos_mask = target.byte()
        neg_mask = 1 - pos_mask

        n_pos = int(torch.sum(pos_mask))
        n_neg = int(torch.sum(neg_mask))
        if len(self.pos_pool) >= self.pos_length and len(self.neg_pool) >= self.neg_length and \
                ((self.ratio > 1 and n_neg > 0 and n_neg > n_pos * self.ratio) or
                 (self.ratio < 1 and n_pos > 0 and n_pos > n_neg / self.ratio)):

            if self.ratio > 1:
                n_selected = max(n_pos * self.ratio, 1)

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
                n_selected = max(n_neg / self.ratio, 1)

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

            if n_pos > 0:
                self.pos_pool.extend(list(torch.masked_select(torch.sigmoid(pred[:, 0]), pos_mask).detach().cpu().numpy()))
            if n_neg > 0:
                self.neg_pool.extend(list(torch.masked_select(torch.sigmoid(pred[:, 0]), neg_mask).detach().cpu().numpy()))
            self.pos_pool = self.pos_pool[-self.pos_length:]
            self.neg_pool = self.neg_pool[-self.neg_length:]

            pos_mean = 1 - self.pos_pool.mean()
            neg_mean = self.neg_pool.mean()
            self.distance = neg_mean - pos_mean
            if abs(self.distance) < 0.1:
                self.ratio = 1
            else:
                self.ratio = math.exp((1 - abs(self.distance)) * 0.67)
                self.ratio = self.ratio if self.distance > 0 else 1 / self.ratio
            # np_contrast = anp / app
            return masked_loss.mean()  # , np_contrast
        else:
            # anp = torch.masked_select(pred[:, 0], neg_mask).mean()
            # app = torch.masked_select(pred[:, 1], pos_mask).mean()
            # np_contrast = anp / app
            # return F.cross_entropy(pred, target)  # , np_contrast
            return F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float())


def reverse_ohem_loss(pred, target, ratio=3): return ohem_loss(pred, target, ratio, reverse=True)


def get_categorial_loss(loss):
    if loss == 'cross_entropy':
        return F.cross_entropy
    elif loss == 'ohem':
        return Ohem, ohem_loss
    elif loss == 'focal':
        return focal_loss
    else:
        raise Exception("Loss '{}' is not supported".format(loss))


def get_categorial_scale(loss):

    scales = [(10263+2032)/16436, (19092+3243)/6396, (26284+991)/1456, (21674+422)/6635, (20991+1947)/5793,
                (13339+1879)/13513, (26200+273)/2258, (14120+10369)/4242, (18731+7585)/2415, (8168+10010)/10553,
                (18275+7571)/2885, (26622+1101)/1008, (19045+1252)/8434, (26507+229)/1995]
    result = []
    for scale in scales:
        # result.append(1/(1+scale))
        # if 0 <= scale < 5:
        #     result.append(0.5)
        # elif 5 <= scale < 10:
        #     result.append(1/3)
        # elif 10 <= scale:
        #     result.append(0.25)
        result.append(scale)

    return result


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

