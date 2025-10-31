import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prep_dataLoadert(filename, batch_size):
    df = pd.read_csv(filename)
    y = df["diagnosis"].map({"M": 1 , "B": 0 }).values
    x = df.drop(columns=["id", "diagnosis"]).values


    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y , dtype=torch.long)


    dataset = TensorDataset(x_tensor , y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset , test_dataset = random_split(dataset, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader , test_loader

def prep_data_skleanr(filename):
    df = pd.read_csv(filename)
    y = df["diagnosis"].map({"M": 1 , "B": 0 }).values
    x = df.drop(columns=["id", "diagnosis"]).values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train , x_test , y_train , y_test = sklearn.model_selection.train_test_split(x_scaled, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


if "__main__" == __name__:
    dataloader  = prep_dataLoadert("./../breast-cancer.csv", 1)





