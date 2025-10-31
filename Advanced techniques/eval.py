import numpy as np
from models import *
import torch
from utils import *
from models import *

def eval_autoencoder_linear(laten_dim, test_loader):
    input_latent = laten_dim


    model = nn.Sequential(
        nn.Linear(input_latent, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load("linear_with_auto.pt"))

    autoencoder = Autoencoder_linear(latent_dim=input_latent)
    autoencoder.load_state_dict(torch.load('autoencoder.pt'))

    model.eval()
    autoencoder.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            _, z = autoencoder(x)
            y_mod = model(z).argmax(dim=1)

            all_preds.append(y_mod)
            all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    conf_matrix = torch.zeros(2, 2, dtype=torch.int32)
    for t, p in zip(all_labels, all_preds):
        conf_matrix[t, p] += 1

    print("Confusion Matrix:")
    print(conf_matrix)

    # Optional: compute accuracy
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    print(f"Accuracy: {accuracy:.4f}")

def eval_autoencoder(laten_dim, test_loader):
    input_latent = laten_dim



    autoencoder = Autoencoder_linear(latent_dim=input_latent)
    autoencoder.load_state_dict(torch.load('autoencoder.pt'))

    autoencoder.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            _ ,z = autoencoder(x)
            z = z.argmax(dim=1)

            all_preds.append(z)
            all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    conf_matrix = torch.zeros(2, 2, dtype=torch.int32)
    for t, p in zip(all_labels, all_preds):
        conf_matrix[t, p] += 1

    print("Confusion Matrix:")
    print(conf_matrix)

    # Optional: compute accuracy
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    print(f"Accuracy: {accuracy:.4f}")


def eval_linear(train_loader):
    input_latent = 30


    model = nn.Sequential(
        nn.Linear(input_latent, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load("linear_without_auto.pt"))


    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in train_loader:
            y_mod = model(x).argmax(dim=1)

            all_preds.append(y_mod)
            all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    conf_matrix = torch.zeros(2, 2, dtype=torch.int32)
    for t, p in zip(all_labels, all_preds):
        conf_matrix[t, p] += 1

    print("Confusion Matrix:")
    print(conf_matrix)

    # Optional: compute accuracy
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    print(f"Accuracy: {accuracy:.4f}")








