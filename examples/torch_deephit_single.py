"""A minimal example of how to fit a DeepHitSigle (DeepHit without competing risks) model with a vanilla torch training
loop. The point of this example is to make it simple to use the DeepHit models in other frameworks that are not based
on torchtuples.
"""
from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data._utils.collate as torch_collate

from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import deephit_single

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper


def get_metabrick_train_val_test() -> Tuple[pd.DataFrame]:
    """Get the METABRICK dataset split into a trainin dataframe and a testing dataframe."""
    df_train = metabric.read_df()
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    return df_train, df_test


def preprocess_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[torch.Tensor]:
    """Preprocess the covariates of the training and test set and return a tensor for the
    taining covariates and test covariates.
    """
    cols_standardize = ["x0", "x1", "x2", "x3", "x8"]
    cols_leave = ["x4", "x5", "x6", "x7"]

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    x_train = x_mapper.fit_transform(df_train).astype("float32")
    x_test = x_mapper.transform(df_test).astype("float32")
    return torch.from_numpy(x_train), torch.from_numpy(x_test)


def make_mlp(in_features: int, out_features: int) -> nn.Module:
    """Make a simple torch net"""
    net = nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.1),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.1),
        nn.Linear(32, out_features),
    )
    return net


def collate_fn(batch) -> Tuple[Tensor]:
    """For a dataset x, duration, event, add the deephit rank matrix to the target"""
    batch = torch_collate.default_collate(batch)
    x, duration, event = batch
    rank_mat = deephit_single.pair_rank_mat(duration.numpy(), event.numpy())
    rank_mat = torch.as_tensor(rank_mat)
    return x, duration, event, rank_mat


def main() -> None:
    # Get the metabrick dataset split in a train and test set
    np.random.seed(1234)
    torch.manual_seed(123)
    df_train, df_test = get_metabrick_train_val_test()

    # Preprocess featuers
    x_train, x_test = preprocess_features(df_train, df_test)

    # Preprocess targetts (we end up with two targets: duration and event).
    # Time is discretized into `num_duration` equidistant points.
    num_durations = 10
    labtrans = deephit_single.LabTransDiscreteTime(num_durations)
    y_train = labtrans.fit_transform(df_train.duration.values, df_train.event.values)
    y_train_duration = torch.from_numpy(y_train[0])
    y_train_event = torch.from_numpy(y_train[1])

    # Make an MLP nerual network
    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    net = make_mlp(in_features, out_features)

    batch_size = 256
    epochs = 20

    # Make dataloader for training set.
    # Note that we want the output of the dataloader to be (x, durations, events, rank_mat), so we need to create
    # the rank_mat in the collate_fn
    train_dataset = TensorDataset(x_train, y_train_duration, y_train_event)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    # Set optimizer and loss function (optimization criterion)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_func = deephit_single.DeepHitSingleLoss(0.2, 0.1)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            x, duration, event, rank_mat = data
            optimizer.zero_grad()
            output = net(x)
            loss = loss_func(output, duration, event, rank_mat)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch: {epoch} -- loss: {running_loss / i}")

    # Predict survival for the test set
    # Set net in evaluation mode and turn off gradients
    net.eval()
    with torch.set_grad_enabled(False):
        output = net(x_test)
        surv = deephit_single.output2surv(output)
    surv_df = pd.DataFrame(surv.numpy().transpose(), labtrans.cuts)

    # Pring the test set concordance index
    ev = EvalSurv(surv_df, df_test.duration.values, df_test.event.values)
    print(f"Concordance: {ev.concordance_td()}")


if __name__ == "__main__":
    main()
