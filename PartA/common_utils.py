### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

dropout_rate = 0.2
test_size = 0.3
seed = 69
batch_size = 128
learning_rate = 0.001
no_folds = 5


def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df["label"] = label_encoder.fit_transform(df["label"])

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    df_train2 = df_train.drop(columns_to_drop, axis=1)
    y_train2 = df_train["label"].to_numpy()

    df_test2 = df_test.drop(columns_to_drop, axis=1)
    y_test2 = df_test["label"].to_numpy()

    return df_train2, y_train2, df_test2, y_test2


def preprocess_dataset(df_train, df_test):
    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled


def set_seed(seed=0):
    """
    set random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits


loss_fn = nn.BCELoss()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss, train_correct = 0, 0

    for _batch, (X, y) in enumerate(dataloader, 0):
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += ((pred > 0.5).float().squeeze() == y.float()).sum().item()

    train_loss /= size
    train_correct /= size

    return train_loss, train_correct


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, test_correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(), y).item()
            test_correct += ((pred > 0.5).float().squeeze() == y.float()).sum().item()

    test_loss /= size
    test_correct /= size

    return test_loss, test_correct


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self._transform(self.X[idx]), self._transform(self.y[idx])

    def _transform(self, data):
        return torch.tensor(data, dtype=torch.float)


def generate_cv_folds_for_batch_sizes(parameters, X_train, y_train):
    """
    returns:
    X_train_scaled_dict(dict) where X_train_scaled_dict[batch_size] is a list of the preprocessed training matrix for the different folds.
    X_val_scaled_dict(dict) where X_val_scaled_dict[batch_size] is a list of the processed validation matrix for the different folds.
    y_train_dict(dict) where y_train_dict[batch_size] is a list of labels for the different folds
    y_val_dict(dict) where y_val_dict[batch_size] is a list of labels for the different folds
    """

    kf = KFold(n_splits=no_folds, shuffle=True, random_state=seed)

    X_train_scaled_dict = {}
    y_train_dict = {}
    X_val_scaled_dict = {}
    y_val_dict = {}

    for batch_size in parameters:
        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []

        for _exp, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_train_folds, y_train_folds = X_train[train_index], y_train[train_index]
            X_val_folds, y_val_folds = X_train[val_index], y_train[val_index]

            X_train_folds_scaled, X_val_folds_scaled = preprocess_dataset(
                X_train_folds, X_val_folds
            )
            X_train_list.append(X_train_folds_scaled)
            y_train_list.append(y_train_folds)
            X_val_list.append(X_val_folds_scaled)
            y_val_list.append(y_val_folds)

        X_train_scaled_dict[batch_size] = X_train_list
        y_train_dict[batch_size] = y_train_list
        X_val_scaled_dict[batch_size] = X_val_list
        y_val_dict[batch_size] = y_val_list

    return X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict
