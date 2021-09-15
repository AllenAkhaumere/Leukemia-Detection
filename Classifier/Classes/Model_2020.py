import numpy as np
import torch
import torch.nn as nn
from Classes.Helpers import Helpers
from collections import defaultdict
from torchvision.models import resnet50

helper = Helpers("Model", False)


class LuekemiaNet(nn.Module):
    def __init__(self):
        super(LuekemiaNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.25),
            nn.AvgPool2d(2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.25),
            nn.AvgPool2d(2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.25),
            nn.AvgPool2d(2))


        self.fc = nn.Sequential(
            nn.Linear(128 * 10 * 10, 200),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(200, 2))

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
def train_epoch(
        model,
        data_loader,
        loss_function,
        optimizer,
        scheduler,
        trainset_size,
        device):

    model = model.train()

    losses = list()
    correct_predictions = 0

    for i, dataset in enumerate(data_loader):
        inputs, labels = dataset
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    scheduler.step()

    return correct_predictions.float() / trainset_size, np.mean(losses)


def eval_model(
        model,
        data_loader,
        loss_function,
        validset_size,
        device
):
    model = model.eval()
    losses = list()
    correct_predictions = 0

    with torch.no_grad():
        for _, dataset in enumerate(data_loader):
            inputs, labels = dataset

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.float() / validset_size, np.mean(losses)


def train_model(model,
                train_data_loader,
                valid_data_loader,
                optimizer,
                scheduler,
                loss_function,
                trainset_size,
                validset_size,
                device,
                n_epochs=10):
    """
    :param model:
    :param train_data_loader:
    :param valid_data_loader:
    :param optimizer:
    :param scheduler:
    :param loss_function:
    :param trainset_size:
    :param validset_size:
    :param train_on_gpu:
    :param n_epochs:
    :return:
    """

    history = defaultdict(list)

    best_accuracy = 0

    for epoch in range(n_epochs):

        helper.logger.info(f'Epoch {epoch + 1}/{n_epochs}')
        helper.logger.info('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_function,
            optimizer,
            scheduler,
            trainset_size,
            device
        )

        helper.logger.info(f'Train loss: {train_loss} Accuracy: {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            valid_data_loader,
            loss_function,
            validset_size,
            device
        )

        helper.logger.info(f'Valid loss: {val_loss} Accuracy: {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), helper.config["classifier"]["model_params"]["weights"])
            best_accuracy = val_acc

    helper.logger.info(f'Best val accuracy: {best_accuracy}')

    model.load_state_dict(torch.load(helper.config["classifier"]["model_params"]["weights"]))

    return model, history






