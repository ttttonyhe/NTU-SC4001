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

# ---------------------- #

# Network
dropout_rate = 0.2
test_size = 0.3
seed = 69
batch_size = 256
learning_rate = 0.001


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


# model = MLP(no_features=77, no_hidden=128, no_labels=1)
# optimizer = torch.optim.Adam(model.parameters(), learning_rate)
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


# Custom dataset and dataloader
# def preprocess(df):
#     X_train, y_train, X_test, y_test = split_dataset(
#         df, ["filename", "label"], test_size, seed
#     )
#     X_train_scaled, X_test_scaled = preprocess_dataset(X_train, X_test)
#     return X_train_scaled, y_train, X_test_scaled, y_test


# df = pd.read_csv("simplified.csv")
# df["label"] = df["filename"].str.split("_").str[-2]
# df["label"].value_counts()
# X_train_scaled, y_train, X_test_scaled, y_test = preprocess(df)


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


# def initialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test):
#     training_data = CustomDataset(X_train_scaled, y_train)
#     testing_data = CustomDataset(X_test_scaled, y_test)
#     train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
#     return train_dataloader, test_dataloader


# train_dataloader, test_dataloader = initialise_loaders(
#     X_train_scaled, y_train, X_test_scaled, y_test
# )
