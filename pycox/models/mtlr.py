
from pycox import models
import torchtuples as tt
from pycox.models import utils

class MTLR(models.pmf.PMFBase):
    """
    The (Neural) Multi-Task Logistic Regression, MTLR [1] and N-MTLR [2].
    A discrete-time survival model that minimize the likelihood for right-censored data.

    This is essentially a PMF parametrization with an extra cumulative sum, as explained in [3].

    Arguments:
        net {torch.nn.Module} -- A torch module.
    
    Keyword Arguments:
        optimizer {Optimizer} -- A torch optimizer or similar. Preferably use torchtuples.optim instead of
            torch.optim, as this allows for reinitialization, etc. If 'None' set to torchtuples.optim.AdamW.
            (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').
        duration_index {list, np.array} -- Array of durations that defines the discrete times.
            This is used to set the index of the DataFrame in `predict_surv_df`.
    
    References:
    [1] Chun-Nam Yu, Russell Greiner, Hsiu-Chin Lin, and Vickie Baracos.
        Learning patient- specific cancer survival distributions as a sequence of dependent regressors.
        In Advances in Neural Information Processing Systems 24, pages 1845–1853.
        Curran Associates, Inc., 2011.
        https://papers.nips.cc/paper/4210-learning-patient-specific-cancer-survival-distributions-as-a-sequence-of-dependent-regressors.pdf

    [2] Stephane Fotso. Deep neural networks for survival analysis based on a multi-task framework.
        arXiv preprint arXiv:1801.05512, 2018.
        https://arxiv.org/pdf/1801.05512.pdf

    [3] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None):
        if loss is None:
            loss = models.loss.NLLMTLRLoss()
        super().__init__(net, loss, optimizer, device, duration_index)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        preds = utils.cumsum_reverse(preds, dim=1)
        pmf = utils.pad_col(preds).softmax(1)[:, :-1]
        return tt.utils.array_or_tensor(pmf, numpy, input)
