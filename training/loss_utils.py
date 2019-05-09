import torch
import torch.nn.functional as F


# alpha now only support for binary classification
# TODO Change it to class so that gamma can also be learned
def focal_loss(pred, target, gamma=2, alpha=None, size_average=True):
    if isinstance(alpha, (float, int)):
        alpha = torch.Tensor([alpha, 1 - alpha])
    if isinstance(alpha, list):
        alpha = torch.Tensor(alpha)

    target = target.view(-1, 1)
    logpt = F.log_softmax(pred, 1)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = logpt.exp()

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
    assert pred.size()[1] == 2  # Only support binary case

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

        ce_loss = F.cross_entropy(pred, target, reduction='none')

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
        return masked_loss.mean()
    else:
        return F.cross_entropy(pred, target)


def reverse_ohem_loss(pred, target, ratio=3): return ohem_loss(pred, target, ratio, reverse=True)


def get_categorial_loss(loss):
    if loss == 'cross_entropy':
        return F.cross_entropy
    elif loss == 'ohem':
        return ohem_loss
    elif loss == 'focal':
        return focal_loss
    else:
        raise Exception("Loss '{}' is not supported".format(loss))


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

    return total_loss
