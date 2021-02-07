'''
A minimal example of how to fit a LogisticHazard model with pytorch lightning
The point of this example is to make it simple to use the LogisticHazard models in other frameworks
that are not based on torchtuples.

Original author: Rohan Shad @rohanshad
'''
from typing import Tuple

import numpy as np
import pandas as pd

import torch

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import logistic_hazard

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

# Lightning Dataset Module


class MetaBrick(pl.LightningDataModule):
	def __init__(self, batch_size: int = 256, num_durations: int = 10, num_workers: int = 0):
		super().__init__()
		self.batch_size = batch_size
		self.num_durations = num_durations
		self.num_workers = num_workers

	def setup(self, stage=None):
		'''
		Get the METABRICK dataset split into a training dataframe and a testing dataframe.
		Preprocesses features and targets (duration and event), discretize time into 'num_duration' equidistant points.
		'''

		# Load and split dataset into train and test (if there's train and val this can be called within stage == 'fit')
		df_train = metabric.read_df()
		df_test = df_train.sample(frac=0.2)
		df_train = df_train.drop(df_test.index)

		self.x_train, self.x_test = self._preprocess_features(df_train, df_test)
		self.labtrans = logistic_hazard.LabTransDiscreteTime(self.num_durations)

		if stage == 'fit' or stage is None:
			# Pre-process features and targets
			self.y_train = self.labtrans.fit_transform(
			    df_train.duration.values, df_train.event.values)
			self.y_train_duration = torch.from_numpy(self.y_train[0])
			self.y_train_event = torch.from_numpy(self.y_train[1])

			# Create training dataset
			self.train_set = TensorDataset(
			    self.x_train, self.y_train_duration, self.y_train_event)

			# Input and output dimensions for building net
			self.in_dims = self.x_train.shape[1]
			self.out_dims = self.labtrans.out_features

		if stage == 'test' or stage is None:
			# Return test dataframe
			self.df_test = df_test

	def train_dataloader(self):
		'''
		Build training dataloader
		num_workers set to 0 by default because of some thread issue
		'''
		train_loader = DataLoader(
			dataset=self.train_set,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers
			)
		return train_loader

	@classmethod
	def _preprocess_features(cls, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[torch.Tensor]:
		'''
		Preprocess the covariates of the training and test set and return a tensor for the
		taining covariates and test covariates.
		'''
		cols_standardize = ["x0", "x1", "x2", "x3", "x8"]
		cols_leave = ["x4", "x5", "x6", "x7"]

		standardize = [([col], StandardScaler()) for col in cols_standardize]
		leave = [(col, None) for col in cols_leave]
		x_mapper = DataFrameMapper(standardize + leave)

		x_train = x_mapper.fit_transform(df_train).astype("float32")
		x_test = x_mapper.transform(df_test).astype("float32")
		return torch.from_numpy(x_train), torch.from_numpy(x_test)

# Survival model class


class SurvModel(pl.LightningModule):
	'''
	Defines model, optimizers, forward step, and training step. 
	Define validation step as def validation_step if needed
	Configured to use NLL logistic hazard loss from logistic_hazard.NLLLogisticHazardLoss()
	'''

	def __init__(self, lr, in_features, out_features):
		super().__init__()

		self.save_hyperparameters()
		self.lr = lr
		self.in_features = in_features
		self.out_features = out_features

		# Define Model Here (in this case MLP)
		self.net = nn.Sequential(
			nn.Linear(self.in_features, 32),
			nn.ReLU(),
			nn.BatchNorm1d(32),
			nn.Dropout(0.1),
			nn.Linear(32, 32),
			nn.ReLU(),
			nn.BatchNorm1d(32),
			nn.Dropout(0.1),
			nn.Linear(32, self.out_features),
		)

		# Define loss function:
		self.loss_func = logistic_hazard.NLLLogistiHazardLoss()

	def forward(self, x):
		batch_size, data = x.size()
		x = self.net(x)
		return x

	# Training step and validation step usually defined, this dataset only had train + test so left out val. 
	def training_step(self, batch, batch_idx): 
		x, duration, event = batch
		output = self.forward(x)
		loss = self.loss_func(output, duration, event)

		# progress bar logging metrics (add custom metric definitions later if useful?)
		self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(
			self.parameters(),
			lr = self.lr
		)
		return optimizer

def main():
	# Load Lightning DataModule
	dat = MetaBrick(num_workers=0)
	dat.setup('fit') #allows for input / output features to be configured in the model

	# Load Lightning Module
	model = SurvModel(lr=1e-3, in_features=dat.in_dims, out_features=dat.out_dims)
	trainer = pl.Trainer(gpus=0, num_sanity_val_steps=0, max_epochs=20, fast_dev_run=False)	
	
	# Train model
	trainer.fit(model,dat)
	
	# Load final model & freeze
	print('Running in Evaluation Mode...')
	model.freeze()

	# Setup test data (prepared from lightning module)
	dat.setup('test')

	# Predict survival on testing dataset
	output = model(dat.x_test)
	surv = logistic_hazard.output2surv(output)
	surv_df = pd.DataFrame(surv.numpy().transpose(), dat.labtrans.cuts)
	ev = EvalSurv(surv_df, dat.df_test.duration.values, dat.df_test.event.values)

	# Print evaluation metrics
	print(f"Concordance: {ev.concordance_td()}")

if __name__ == '__main__':
	main()