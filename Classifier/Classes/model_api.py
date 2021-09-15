import numpy as np
import PIL.Image as Image
import torch
from Classifier.Classes.visualizer import image_show
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(history['train_loss'], label='train loss')
    ax1.plot(history['val_loss'], label='validation loss')

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    ax2.plot(history['train_acc'], label='train accuracy')
    ax2.plot(history['val_acc'], label='validation accuracy')

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend()

    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')

    fig.suptitle('Training history')
    plt.savefig(save_path)
    plt.show()


def show_predictions(model,
                     class_names,
                     test_data_loader,
                     use_gpu=True,
                     n_images=6):
    model = model.eval()
    model.cuda()
    images_handeled = 0
    plt.figure()

    with torch.no_grad():
        for i, dataset in enumerate(test_data_loader):
            inputs, labels = dataset
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.shape[0]):
                images_handeled += 1
                ax = plt.subplot(2, n_images // 2, images_handeled)
                ax.set_title(f'predicted:{class_names[preds[j]]} ')
                image_show(inputs.cpu().data[j])
                ax.axis('off')
                if images_handeled == n_images:
                    return


def get_predictions(model, data_loader, device):
    model = model.eval()
    y_predictions = list()
    y_true = list()
    with torch.no_grad():
        for i, dataset in enumerate(data_loader):
            inputs, labels = dataset
            inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_predictions.extend(preds)
        y_true.extend(labels)

    predictions = torch.as_tensor(y_predictions).cpu()
    y_true = torch.as_tensor(y_true).cpu()
    return predictions, y_true


def confusion_matrix2(confusion_matrix, class_names, save_path):
    cm = confusion_matrix.copy()

    cell_counts = cm.flatten()

    cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]

    row_percentages = ["{0:.2f}".format(value) for value in cm_row_norm.flatten()]

    cell_labels = [f"{cnt}\n{per}" for cnt, per in zip(cell_counts, row_percentages)]
    cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])

    df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)

    hmap = sns.heatmap(df_cm, annot=cell_labels, fmt="", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True diagnostic')
    plt.xlabel('Predicted diagnostic')
    plt.savefig(save_path)
    plt.show()


def predict_probability(model, transforms, image_path, use_gpu=True):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = transforms(img).unsqueeze(0)
    if use_gpu:
        image = img.cuda()

    predictions = model(image)
    predictions = torch.sigmoid(predictions)
    predictions = predictions.detach().cpu().numpy().flatten()
    return predictions


def show_prediction_confidence(prediction, class_names):
    pred_df = pd.DataFrame({
        'class_names': class_names,
        'values': prediction
    })
    sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
    sns
    plt.xlim([0, 1])

