'''
A minimal example of how to fit a LogisticHazard model with a vanilla torch training loop.
The point of this example is to make it simple to use the LogisticHazard models in other frameworks
that are not based on torchtuples.
'''
from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import coxph

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt

def get_metabrick_train_val_test() -> Tuple[pd.DataFrame]:
	"""Get the METABRICK dataset split into a trainin dataframe and a testing dataframe."""
	df_train = metabric.read_df()
	df_test = df_train.sample(frac=0.2)
	df_train = df_train.drop(df_test.index)
	df_val = df_train.sample(frac=0.2)
	df_train = df_train.drop(df_val.index)
	return df_train, df_val, df_test


def preprocess_features(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[torch.Tensor]:
	"""Preprocess the covariates of the training and test set and return a tensor for the
	taining covariates and test covariates.
	"""
	cols_standardize = ["x0", "x1", "x2", "x3", "x8"]
	cols_leave = ["x4", "x5", "x6", "x7"]

	standardize = [([col], StandardScaler()) for col in cols_standardize]
	leave = [(col, None) for col in cols_leave]
	x_mapper = DataFrameMapper(standardize + leave)

	x_train = x_mapper.fit_transform(df_train).astype("float32")
	x_val = x_mapper.transform(df_val).astype("float32")
	x_test = x_mapper.transform(df_test).astype("float32")

	return torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(x_test)


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

def get_target(df : pd.DataFrame) -> np.ndarray:
	'''
	Takes pandas datframe and converts the duration / event targets into np.array
	'''
	duration = df['duration'].to_numpy().reshape(len(df['duration']),1)
	event = df['event'].to_numpy().reshape(len(df['event']),1)

	return duration, event

def main() -> None:
	# Get the metabrick dataset split in a train and test set
	#np.random.seed(1234)
	#torch.manual_seed(123)
	df_train, df_val, df_test = get_metabrick_train_val_test()

	# Preprocess features
	x_train, x_val, x_test = preprocess_features(df_train, df_val, df_test)
	y_train = torch.from_numpy(np.concatenate(get_target(df_train), axis=1))
	y_test = torch.from_numpy(np.concatenate(get_target(df_test), axis=1))

	y_val = torch.from_numpy(np.array(get_target(df_val)))
	
	#Probably have to change this to something like test_target?
	durations_test, events_test = get_target(df_test)
	val = x_val, y_val

	# Make an MLP nerual network
	in_features = x_train.shape[1]
	out_features = 1
	net = make_mlp(in_features, out_features)

	batch_size = 256
	epochs = 20

	train_dataset = TensorDataset(x_train, y_train)
	train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

	# if verbose:
	# 	print('Durations and events in order:')
	# 	print(y_train[:,0])
	# 	print(y_train[:,1]

	# Set optimizer and loss function (optimization criterion)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	loss_func = coxph.CoxPHLoss()
	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(train_dataloader):
			x, target = data
			optimizer.zero_grad()
			output = net(x)
			loss = loss_func(output, target[:,0], target[:,1])  # need x, durations, events
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print(f"epoch: {epoch} -- loss: {running_loss / i}")

	# Predict survival for the test set
	# Set net in evaluation mode and turn off gradients


	net.eval()
	with torch.set_grad_enabled(False):
		output = net(x_test)
		#Cumulative Hazards Calculation
		cum_haz = coxph.compute_cumulative_baseline_hazards(output, durations=y_test[:,0], events=y_test[:,1])
	
	surv = coxph.output2surv(output, cum_haz[0])
	# The dataframe needs to be transposed to format: {rows}: duration, {cols}: each individual
	surv_df = pd.DataFrame(surv.transpose(1,0).numpy())
	print(surv_df)
	
	# print the test set concordance index
	ev = EvalSurv(surv_df, df_test.duration.values, df_test.event.values, censor_surv='km')
	time_grid = np.linspace(df_test.duration.values.min(), df_test.duration.values.max(), 100)

	print(f"Concordance: {ev.concordance_td()}")
	print(f"Brier Score: {ev.integrated_brier_score(time_grid)}")


if __name__ == "__main__":
	main()