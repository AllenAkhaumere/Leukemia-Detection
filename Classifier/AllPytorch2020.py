import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from Classes.Data import LeukemiaDataset, augmentation
from Classes.Helpers import Helpers
from Classes.interpretability import interpret_model
from Classes.Model_2020 import LuekemiaNet, train_model
from Classes.model_api import (confusion_matrix2, get_predictions,
                               plot_training_history)

SEED = 323


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


image_path = '/home/allen/Drive C/Peter Moss AML Leukemia Research/ALL-PyTorch-2020/Classifier/Model/Data/Test/Im047_0.jpg'
image_path2 = '/home/allen/Drive C/Peter Moss AML Leukemia Research/ALL-PyTorch-2020/Classifier/Model/Data/Test/Im006_1.jpg'
label_idx = '/home/allen/Drive C/Peter Moss AML Leukemia Research/Dataset/pred_label.json'

seed_everything(SEED)
# helper class
helper = Helpers("Test Model", False)
# train data directory
train_dir = '/home/allen/Drive C/Peter Moss AML Leukemia Research/Dataset/all_train/'
# train label directoy
train_csv = '/home/allen/Drive C/Peter Moss AML Leukemia Research/Dataset/train.csv'
# labels
class_name = ["zero", "one"]

# training batch size
batch_size = helper.config["classifier"]["train"]["batch"]
# accuracy and loss save directory
acc_loss_png = helper.config["classifier"]["model_params"]["plot_loss_and_acc"]
# confusion matrix save directory
confusion_png = helper.config["classifier"]["model_params"]["confusion_matrix"]
# number of epoch
epochs = helper.config["classifier"]["train"]["epochs"]
# learning rate
learn_rate = helper.config["classifier"]["train"]["learning_rate_adam"]
# decay
decay = helper.config["classifier"]["train"]["decay_adam"]
# read train CSV file
labels = pd.read_csv(train_csv)
# print label count
labels_count = labels.label.value_counts()
print(labels_count)
# print 5 label header
print(labels.head())
# splitting data into training and validation set
train, valid = train_test_split(labels, stratify=labels.label, test_size=0.2, shuffle=True)
print(len(train), len(valid))

# data augmentation
training_transforms, validation_transforms = augmentation()

# Read Acute Lymphoblastic Leukemia dataset from disk
trainset = LeukemiaDataset(df_data=train, data_dir=train_dir, transform=training_transforms)
validset = LeukemiaDataset(df_data=valid, data_dir=train_dir, transform=validation_transforms)

train_size, valid_size = len(trainset), len(validset)
print(train_size, valid_size)

train_sampler = SubsetRandomSampler(list(train.index))
valid_sampler = SubsetRandomSampler(list(valid.index))

# Prepare dataset for neural networks
train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_data_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)

# Checking the dataset
print('Training Set:\n')
for images, labels in train_data_loader:
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    break
print("\n")

#
for images, labels in valid_data_loader:
    print("The labels: ", labels)

# Define model
model = LuekemiaNet()
# check if CUDA is available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move our model to available hardware
model.to(device)

# Cross entropy loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = Adam(params=model.parameters(), lr=learn_rate, weight_decay=decay)
# scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=0.01, step_size=5, mode='triangular2')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7,
                                      gamma=helper.config["classifier"]["model_params"]["gamma"])

# begin training
real_model, history = train_model(model, train_data_loader,
                                  valid_data_loader,
                                  optimizer, scheduler,
                                  criterion, train_size,
                                  valid_size,
                                  device=device, n_epochs=epochs)
# plot model loss and accuracy
plot_training_history(history, save_path=acc_loss_png)
# Get model prediction
y_pred, y_test = get_predictions(real_model, valid_data_loader, device)
# Get model precision, recall and f1_score
helper.logger.info(classification_report(y_test, y_pred, target_names=class_name))
# Get model confusion matrix
cm = confusion_matrix(y_test, y_pred)
confusion_matrix2(cm, class_name, save_path=confusion_png)

interpret_model(real_model, validation_transforms, image_path, label_idx, use_cpu=True,
                interpret_type="integrated gradients")
interpret_model(real_model, validation_transforms, image_path, label_idx, use_cpu=True, interpret_type="gradient shap")

interpret_model(real_model, validation_transforms, image_path2, label_idx, use_cpu=True,
                interpret_type="integrated gradients")
interpret_model(real_model, validation_transforms, image_path2, label_idx, use_cpu=True, interpret_type="gradient shap")


