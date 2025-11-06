from models import *
from train import *
from utils import *
from eval import *
print("###########random forest########")
random_forest_train_eval()
print("############# gradient boosting ##########")
gradient_boost_train_eval()

input_latent = 10
batch_size = 32
train_loader, test_loader = prep_dataLoadert("./../breast-cancer.csv", batch_size=batch_size)

autoencoder_train_simple_linear(input_latent, train_loader)
classifer_train(input_latent, train_loader)
classifer_train_without_auto(train_loader)

print("############ autoencoder with linear classifier#############")
eval_autoencoder_linear(input_latent, test_loader)
print("############  linear classifier (MLP)#############")
eval_linear(test_loader)