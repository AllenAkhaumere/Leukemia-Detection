import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Classifier.Classes.Helpers import Helpers

helper = Helpers("Model", False)

class LuekemiaNet(nn.Module):
    """ Model Class

        Model functions for the Acute Lymphoblastic Leukemia Pytorch CNN 2020.
        """
    def __init__(self):
        super(LuekemiaNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2))

        self.fc = nn.Sequential(
            nn.Linear(128*2*2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 2))

    def forward(self, x):
        """Method for Forward Prop"""
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        ######################
        # For model debugging #
        ######################
        #print(out.shape)

        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out



###################
# Train model     #
###################
def learn(model, loss_function, optimizer, epochs, train_loader,
          valid_loader, scheduler, train_on_gpu=True):
    """ """

    trainset_auc = list()
    train_losses = list()
    valid_losses = list()
    val_auc = list()
    valid_auc_epoch = list()
    train_auc_epoch = list()

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train_model()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # first of all, set gradient of all optimized variables to zero
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs data to the model
            outputs = model(inputs)
            # compute batch loss(binary cross entropy
            loss = loss_function(outputs[:,0], labels.float())
            # Update Train loss and accuracies
            train_loss += loss.item() * inputs.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform one step of optimization
            optimizer.step()
            scheduler.step()
            y_actual = labels.data.cpu().numpy()
            prediction = outputs[:, 0].detach().cpu().numpy()
            trainset_auc.append(roc_auc_score(y_true=y_actual, y_score=prediction))

        model.eval()
        for _, data in enumerate(valid_loader):
            inputs, labels = data
            # if CUDA GPU is available
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
                # first of all, set gradient of all optimized variables to zero
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs data to the model
            outputs = model(inputs)
            # compute batch loss(binary cross entropy
            loss = loss_function(outputs[:,0], labels.float())
            # update average validation loss
            valid_loss += loss.item() * inputs.size(0)
            y_actual = labels.data.cpu().numpy()
            predictionx = outputs[:, 0].detach().cpu().numpy()
            val_auc.append(roc_auc_score(y_actual, predictionx))

            # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        valid_auc = np.mean(val_auc)
        train_auc = np.mean(trainset_auc)
        train_auc_epoch.append(np.mean(trainset_auc))
        valid_auc_epoch.append(np.mean(val_auc))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        helper.logger.info(
            'Epoch: {} | Training Loss: {:.6f} | Training AUC: {:.6f}| Validation Loss: {:.6f} | Validation AUC: {:.4f}'.format(
                    epoch, train_loss, train_auc, valid_loss, valid_auc))

    torch.save(model.state_dict(), helper.config["classifier"]["model_params"]["weights"])

    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.savefig(helper.config["classifier"]["model_params"]["plot_loss"])
    plt.show()

    plt.plot(valid_auc_epoch, label='Validation AUC/Epochs')
    plt.plot(train_auc_epoch, label='Training AUC/Epochs')
    plt.legend("")
    plt.xlabel("Epochs")
    plt.ylabel("Area Under the Curve")
    plt.legend(frameon=False)
    plt.savefig(helper.config["classifier"]["model_params"]["plot_auc"])
    plt.show()

