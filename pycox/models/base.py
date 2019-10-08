import torchtuples as tt


class _SurvModelBase(tt.Model):
    """Base class for survival models. 
    Essentially same as torchtuples.Model, 
    """
    def __init__(self, net, loss=None, optimizer=None, device=None):
        super().__init__(net, loss, optimizer, device)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the survival function for `input`.
        See `prediction_surv_df` to return a DataFrame instead.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        
        Returns:
            [TupleTree, np.ndarray or tensor] -- Predictions
        """
        raise NotImplementedError

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        """Predict the survival function for `input` and return as a pandas DataFrame.
        See `predict_surv` to return tensor or np.array instead.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        
        Returns:
            pd.DataFrame -- Predictions
        """
        raise NotImplementedError
