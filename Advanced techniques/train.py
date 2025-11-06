import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from models import *
import torch
from utils import *



def autoencoder_train_simple_linear(input_latent, train_loader):


    num_epochs = 200
    autoencoder = Autoencoder_linear(latent_dim=input_latent)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # map reconstruction loss

    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            x , _ = data
            #x = x.view(x.size(0),  -1)

            optimizer.zero_grad()

            recon_x, _ = autoencoder(x)
            loss = criterion(recon_x, x)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")
    torch.save(autoencoder.state_dict(), 'autoencoder.pt')


def classifer_train(input_latent, train_loader):
    model = nn.Sequential(
        nn.Linear(input_latent, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.LogSoftmax(dim=1)
    )
    batch_size = 32


    num_epochs = 200
    autoencoder = Autoencoder_linear(latent_dim=input_latent)
    autoencoder.load_state_dict(torch.load('autoencoder.pt'))




    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()  # map reconstruction loss

    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            x, y = data

            optimizer.zero_grad()

            _, latent = autoencoder(x)
            y_mod = model(latent)
            # y_mod = y_mod.argmax(dim=1)
            loss = criterion(y_mod, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")
    torch.save(model.state_dict(), 'linear_with_auto.pt')

def classifer_train_without_auto(train_loader):
    model = nn.Sequential(
        nn.Linear(30, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.LogSoftmax(dim=1)
    )
    batch_size = 32


    num_epochs = 200





    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()  # map reconstruction loss

    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            x, y = data

            optimizer.zero_grad()

            y_mod = model(x)
            # y_mod = y_mod.argmax(dim=1)
            loss = criterion(y_mod, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")
    torch.save(model.state_dict(), 'linear_without_auto.pt')

def random_forest_train_eval():
    x_train, x_test, y_train, y_test = prep_data_skleanr("./../breast-cancer.csv")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(conf_matrix)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))

    cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.title("Random Forest Confusion Matrix")
    plt.show()

def gradient_boost_train_eval():
    x_train, x_test, y_train, y_test = prep_data_skleanr("./../breast-cancer.csv")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42)
    gb.fit(x_train, y_train)

    y_pred = gb.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(conf_matrix)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))

    cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.title("Gradient Boosting Confusion Matrix")
    plt.show()





if "__main__" == __name__:
    random_forest_train_eval()
    gradient_boost_train_eval()

    input_latent = 2
    batch_size = 32
    train_loader , test_loader  = prep_dataLoadert("./../breast-cancer.csv", batch_size=batch_size)

    autoencoder_train_simple_linear(input_latent, train_loader)
    classifer_train(input_latent , train_loader)
    classifer_train_without_auto(train_loader)