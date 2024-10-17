import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm, trange


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 32)
        self.hidden_fc = nn.Linear(32, 16)
        self.output_fc = nn.Linear(16, output_dim)

    def forward(self, x):

        # x = [batch size, height * width]

        h_1 = torch.sigmoid(self.input_fc(x))

        # h_1 = [batch size, 32]

        h_2 = torch.sigmoid(self.hidden_fc(h_1))

        # h_2 = [batch size, 16]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return torch.sigmoid(torch.squeeze(y_pred, dim=1)), h_2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y_true):

    y_true = torch.squeeze(y_true).cpu().numpy()
    y_pred = torch.squeeze(y_pred).detach().cpu().numpy()

    assert y_true.size == y_pred.size
    y_pred = y_pred > 0.5
    return np.array(y_true == y_pred).sum().item()


def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y_true) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y_true = y_true.to(device)

            y_pred, _ = model(x)

            y_true = torch.squeeze(y_true)
            y_pred = torch.squeeze(y_pred)

            loss = criterion(y_pred, y_true)

            acc = calculate_accuracy(y_pred, y_true)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
