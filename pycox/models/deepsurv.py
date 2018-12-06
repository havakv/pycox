"""An implementation of the DeepSurv paper"""

from lifelines.utils import concordance_index
from pyth import Model, tuplefy, make_dataloader
from pyth.data import DatasetTuple
# from pycox.models.cox_pyth import CoxPH

def loss_deepsurv(risk, event):
    event = event.view(-1)
    risk = risk.view(-1)
    log_risk = risk.exp().cumsum(0).log()
    return - risk.sub(log_risk).mul(event).sum().div(event.sum())

class DatasetDurationSorted(DatasetTuple):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        input, (duration, event) = batch
        idx_sort = duration.sort(descending=True)[1]
        event = event.float()
        batch = tuplefy(input, event).iloc[idx_sort]
        return batch

class DeepSurv(Model):
    def __init__(self, net, optimizer=None, device=None):
        loss = loss_deepsurv
        return super().__init__(net, loss=loss, optimizer=optimizer, device=device)

    @staticmethod
    def make_dataloader(data, batch_size, shuffle, num_workers=0):
        dataloader = make_dataloader(data, batch_size, shuffle, num_workers,
                                     make_dataset=DatasetDurationSorted)
        return dataloader

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def concordance_index(self, input, target):
        preds = self.predict(input)
        durations, events = target
        return 1 - concordance_index(durations, preds, events)