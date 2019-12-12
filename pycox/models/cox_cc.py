import numpy as np
import torchtuples as tt
from pycox import models


class _CoxCCBase(models.cox._CoxBase):
    make_dataset = NotImplementedError

    def __init__(self, net, optimizer=None, device=None, shrink=0., loss=None):
        if loss is None:
            loss = models.loss.CoxCCLoss(shrink)
        super().__init__(net, loss, optimizer, device)

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            n_control=1, shrink=None, **kwargs):
        """Fit  model with inputs and targets. Where 'input' is the covariates, and
        'target' is a tuple with (durations, events).
        
        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.
            target {np.array, tensor or tuple} -- Target [durations, events]. 
        
        Keyword Arguments:
            batch_size {int} -- Elements in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            num_workers {int} -- Number of workers used in the dataloader (default: {0})
            shuffle {bool} -- If we should shuffle the order of the dataset (default: {True})
            n_control {int} -- Number of control samples.
            **kwargs are passed to 'make_dataloader' method.
    
        Returns:
            TrainingLogger -- Training log
        """
        input, target = self._sorted_input_target(input, target)
        if shrink is not None:
            self.loss.shrink = shrink
        return super().fit(input, target, batch_size, epochs, callbacks, verbose,
                           num_workers, shuffle, metrics, val_data, val_batch_size,
                           n_control=n_control, **kwargs)

    def compute_metrics(self, input, metrics):
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
        input = self._to_device(input)
        batch_size = input.lens().flatten().get_if_all_equal()
        if batch_size is None:
            raise RuntimeError("All elements in input does not have the same length.")
        case, control = input # both are TupleTree
        input_all = tt.TupleTree((case,) + control).cat()
        g_all = self.net(*input_all)
        g_all = tt.tuplefy(g_all).split(batch_size).flatten()
        g_case = g_all[0]
        g_control = g_all[1:]
        res = {name: metric(g_case, g_control) for name, metric in metrics.items()}
        return res

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        """Dataloader for prediction. The input is either the regular input, or a tuple
        with input and label.
        
        Arguments:
            input {np.array, tensor, tuple} -- Input to net, or tuple with input and labels.
            batch_size {int} -- Batch size.
        
        Keyword Arguments:
            shuffle {bool} -- If we should shuffle in the dataloader. (default: {False})
            num_workers {int} -- Number of worker in dataloader. (default: {0})
        
        Returns:
            dataloader -- A dataloader.
        """
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader
    
    def make_dataloader(self, data, batch_size, shuffle=True, num_workers=0, n_control=1):
        """Dataloader for training. Data is on the form (input, target), where
        target is (durations, events).
        
        Arguments:
            data {tuple} -- Tuple containing (input, (durations, events)).
            batch_size {int} -- Batch size.
        
        Keyword Arguments:
            shuffle {bool} -- If shuffle in dataloader (default: {True})
            num_workers {int} -- Number of workers in dataloader. (default: {0})
            n_control {int} -- Number of control samples in dataloader (default: {1})
        
        Returns:
            dataloader -- Dataloader for training.
        """
        input, target = self._sorted_input_target(*data)
        durations, events = target
        dataset = self.make_dataset(input, durations, events, n_control)
        dataloader = tt.data.DataLoaderBatch(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
        return dataloader

    @staticmethod
    def _sorted_input_target(input, target):
        input, target = tt.tuplefy(input, target).to_numpy()
        durations, _ = target
        idx_sort = np.argsort(durations)
        if (idx_sort == np.arange(0, len(idx_sort))).all():
            return input, target
        input = tt.tuplefy(input).iloc[idx_sort]
        target = tt.tuplefy(target).iloc[idx_sort]
        return input, target


class CoxCC(_CoxCCBase, models.cox._CoxPHBase):
    """Cox proportional hazards model parameterized with a neural net and
    trained with case-control sampling [1].
    This is similar to DeepSurv, but use an approximation of the loss function.
    
    Arguments:
        net {torch.nn.Module} -- A PyTorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    References:
    [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
        Time-to-event prediction with neural networks and Cox regression.
        Journal of Machine Learning Research, 20(129):1–30, 2019.
        http://jmlr.org/papers/v20/18-424.html
    """
    make_dataset = models.data.CoxCCDataset
