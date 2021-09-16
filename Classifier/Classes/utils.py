import csv
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt




def image_to_csv(file_path, save_path, mode=""):

    """
    Create CSV from Acute Lymphoblastic Leukemia dataset
    Args:
        file_path: Acute Lymphoblastic Leukemia data path
        save_path: path to save the created CSV
        mode:
    """

    columns = ['data', 'label']
    csv_data = ""
    imagefiles = list()  # create a list to store image names
    if mode == "train":
        csv_data = "train.csv"
    if mode == "test":
        csv_data = "test.csv"

    with open(os.path.join(save_path, csv_data), 'w', newline='') as csvfile:
        for root, dirs, files in os.walk(file_path): #scan through the file path
            for file in files: # loop through all files
                if '.tif' or '.jpg' or '.png' in file: #chech if there are files with *.tif, *.jpg or *.png
                    imagefiles.append(os.path.splitext(file)[0])#retrieve file names and add to imagefiles list

        writer = csv.writer(csvfile, dialect='excel')  # Create a writer from csv module
        writer.writerow(columns)#write down the columns
        for image in imagefiles:# loop through all image names in the imagefiles list
            label = os.path.basename(image)
            if "_0" in label:
                label = 0
            elif "_1" in label:
                label = 1

            writer.writerow([image, label])
    print("done")

    




def print_accuracy_and_classification_report(labels, prediction):
    """Print model accuracy and classification report.

    Args:
        labels (numpy.array): Truth label
        prediction (numpy.array): Model predictions
    """
    print('Cross validation accuracy:')
    print('\t', metrics.accuracy_score(labels, prediction))
    print('\nCross validation classification report\n')
    print(metrics.classification_report(labels, prediction))



def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def accuracy_mini_batch(predicted, true, i, acc, tpr, tnr):
    predicted = predicted.cpu()
    true = true.cpu()

    predicted = (sigmoid(predicted.data.numpy()) > 0.5)
    true = true.data.numpy()

    accuracy = np.sum(predicted == true) / true.shape[0]
    true_positive_rate = np.sum((predicted == 1) * (true == 1)) / np.sum(true == 1)
    true_negative_rate = np.sum((predicted == 0) * (true == 0)) / np.sum(true == 0)
    acc = acc * (i) / (i + 1) + accuracy / (i + 1)
    tpr = tpr * (i) / (i + 1) + true_positive_rate / (i + 1)
    tnr = tnr * (i) / (i + 1) + true_negative_rate / (i + 1)

    return acc, tpr, tnr


def accuracy(predicted, true):
    predicted = predicted.cpu()
    true = true.cpu()

    predicted = (sigmoid(predicted.data.numpy()) > 0.5)
    true = true.data.numpy()

    accuracy = np.sum(predicted == true) / true.shape[0]
    true_positive_rate = np.sum((predicted == 1) * (true == 1)) / np.sum(true == 1)
    true_negative_rate = np.sum((predicted == 0) * (true == 0)) / np.sum(true == 0)

    return accuracy, true_positive_rate, true_negative_rate


def model_confusion_matrix(y_true, y_pred, classes=[]):

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    hmap = sns.heatmap(df_cm, annot=True, fmt='d')
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.show()








                          
