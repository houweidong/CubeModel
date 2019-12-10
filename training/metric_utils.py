import torch
import torch.nn.functional as F
from ignite.metrics import Metric, Loss
from ignite.contrib.metrics import AveragePrecision
from utils.table import TableForPrint
from ignite.exceptions import NotComputableError

# class EpochMetric(Metric):
#     _predictions, _targets = None, None
#
#     def reset(self):
#         self._predictions = torch.tensor([], dtype=torch.float32)
#         self._targets = torch.tensor([], dtype=torch.long)
#
#     def update(self, output):
#         y_pred, y = output
#
#         assert 1 <= y_pred.ndimension() <= 2, "Predictions should be of shape (batch_size, n_classes)"
#         assert 1 <= y.ndimension() <= 2, "Targets should be of shape (batch_size, n_classes)"
#
#         if y.ndimension() == 2:
#             assert torch.equal(y ** 2, y), 'Targets should be binary (0 or 1)'
#
#         if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
#             y_pred = y_pred.squeeze(dim=-1)
#
#         if y.ndimension() == 2 and y.shape[1] == 1:
#             y = y.squeeze(dim=-1)
#
#         y_pred = y_pred.type_as(self._predictions)
#         y = y.type_as(self._targets)
#
#         self._predictions = torch.cat([self._predictions, y_pred], dim=0)
#         self._targets = torch.cat([self._targets, y], dim=0)
#
#     @abstractmethod
#     def compute(self):
#         pass
#
# class AveragePrecision(EpochMetric):
#     def __init__(self, reverse=False):
#         super().__init__()
#         self.reverse = reverse
#
#     def compute(self):
#         y_true = self._targets.numpy()
#         y_pred = F.softmax(self._predictions, 1).numpy()
#         if not self.reverse:
#             return average_precision_score(y_true, y_pred[:, 1])
#         else:
#             # Treat negative example as positive and vice-verse, to calculate AP
#             return average_precision_score(1 - y_true, y_pred[:, 0])


class MyAccuracy(Metric):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        y_pred = self._output_transform(y_pred)

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred "
                             "must have shape of (batch_size, num_classes, ...) or (batch_size, ...).")

        if y.ndimension() > 1 and y.shape[1] == 1:
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if y_pred.ndimension() == y.ndimension():
            # Maps Binary Case to Categorical Case with 2 classes
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        indices = torch.max(y_pred, dim=1)[1]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class MultiAttributeMetric(Metric):
    def __init__(self, metrics_per_attr, tasks):
        self.names = tasks
        self.metrics_per_attr = [ma if isinstance(ma, list) else [ma] for ma in metrics_per_attr]
        super().__init__()

    def reset(self):
        for ma in self.metrics_per_attr:
            for m in ma:
                m.reset()

    def update(self, output):
        # self.names jinshen_yesno, jinshen_recog,        kuzijinshen_yesno, kuzijinshen_recog
        # preds:     attr1 logits,  attr1_recog logits,   attr2 logits,      attr2_recog logits,  attr1_at, attr2_at
        # so match the front 4 items of preds use lenght of self.names, and discard the last two at predits
        preds, (target, mask) = output
        for i in range(len(self.names)):
            if mask[i].any():
                pred = torch.masked_select(preds[i], mask[i]).view(-1, preds[i].size()[1])
                gt = torch.masked_select(target[i], mask[i])
                # pred, gt = select_samples_by_mask(preds[i], target[i], mask[i], index)
                for m in self.metrics_per_attr[i]:
                    m.update((pred, gt))

    def compute(self):
        # logger_print = create_orderdict_for_print()
        table = TableForPrint()

        # for each Attribute:
        for name, ma in zip(self.names, self.metrics_per_attr):
            table.reset(name)
            # Set metric display name
            for m in ma:
                m_name = self.print_metric_name(m)
                table.update(name, m_name, m.compute())
        table.summarize()
        return {'metrics': table.metrics, 'summaries': table.summary, 'logger': table.logger_print}

    @staticmethod
    def print_metric_name(metric):
        if isinstance(metric, AveragePrecision):
            return 'ap'
        elif isinstance(metric, MyAccuracy):
            return 'accuracy'
        elif isinstance(metric, Loss):
            return 'loss'
        else:
            return metric.__class__.__name__.lower()


# Utility Metric to return a scaled output of the actual metric
# class ScaledError(Metric):
#     def __init__(self, metric, scale=1.0):
#         self.metric = metric
#         self.scale = scale
#
#         super().__init__()
#
#     def reset(self):
#         self.metric.reset()
#
#     def update(self, output):
#         self.metric.update(output)
#
#     def compute(self):
#         return self.scale * self.metric.compute()
