import torch
import torch.nn.functional as F
from ignite.metrics import Metric, CategoricalAccuracy, Loss, Accuracy
from ignite.contrib.metrics import AveragePrecision
from utils.table import TableForPrint


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

class MultiAttributeMetric(Metric):
    def __init__(self, metrics_per_attr, tasks):
        self.attrs, self.names = tasks
        self.metrics_per_attr = [ma if isinstance(ma, list) else [ma] for ma in metrics_per_attr]
        super().__init__()

    def reset(self):
        for ma in self.metrics_per_attr:
            for m in ma:
                m.reset()

    def update(self, output):
        preds, (target, mask) = output
        n_tasks = len(target)
        for i in range(n_tasks):
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
        elif isinstance(metric, Accuracy):
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
