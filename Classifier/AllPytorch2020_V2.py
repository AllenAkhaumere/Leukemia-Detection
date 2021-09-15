import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.optim import Adam
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import PIL.Image as Image

import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import NoiseTunnel
from captum.attr import Saliency
from captum.attr import visualization as viz


SEED = 323
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#decay
decay = 5e-4
# training batch size
batch_size = 10
# check if cuda is available: if not available, then use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeukemiaDataset(Dataset):

    """
    Acute Lymphoblastic Leukemia Dataset Reader.

     Args:
           df_data: Dataframe for CSV file
           data_dir: path to Lymphoblastic Leukemia Data
           transform: transforms for performing data augmentation
    """

    def __init__(self, df_data, data_dir='./', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name + '.jpg')
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def augmentation():
    """Acute Lymphoblastic Leukemia data augmentation"""
    mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    training_transforms = transforms.Compose([transforms.Resize((100, 100)),
                                              transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(100),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomGrayscale(p=0.1),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std_dev)])

    validation_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)])

    return training_transforms, validation_transforms




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

def get_predictions(model, data_loader, use_gpu=True):
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


color = [(0, '#ffffff'), (0.25, '#000000'), (1, '#000000')]
name = 'custom blue'
N = 256

def linear_seg_color_map(name, color, N, gamma=1.0):
    """
    Render color map based on lookup tables
    :param name: name the color
    :param color: color code
    :param N: number of RGB quantization
    :param gamma: default is 1.0
    :return:
    """
    default_cmap = LinearSegmentedColormap.from_list(name, color, N, gamma)
    return default_cmap

def predict(model, transforms, image_path, use_cpu):
    """

    :param model:
    :param transforms: data transform
    :param image_path: inference input image path
    :param use_cpu:
    :return:
    """
    model.cpu()
    model = model.eval()
    img = Image.open(image_path)
    img = img.convert('RGB')
    transformed_img = transforms(img)
    image = transformed_img
    image = image.unsqueeze(0)
    if use_cpu:
        image = image.cpu()
    elif use_cpu == False:
        model.cuda()
        image = image.cuda
    output = model(image)

    output = torch.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    return image, transformed_img, prediction_score, pred_label_idx

def interpret_model(model, transforms, image_path, label_path, use_cpu=True, interpret_type=""):

    """
    :param model: our model
    :param transforms: Data transformation
    :param image_path: Image directory
    :param label_path: Json label directory
    :param use_gpu: set gpu to True
    :param interpret_type: mode for model interpretability: "integrated gradients"
    for Integrated Gradients, "gradient shap" for Gradient Shap and "occlusion" for Occlusion
    :return:
    """
    with open(label_path) as json_data:
        idx_to_labels = json.load(json_data)

    # Check if mode is Integrated Gradients
    if interpret_type == "integrated gradients":

        print('Performing Integrated Gradients Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model, transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(image, target=pred_label_idx, n_steps=20)

        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=linear_seg_color_map(name, color, N),
                                              show_colorbar=True)

    # Check if mode is Integrated Gradients with Noise Tunnel
    elif interpret_type == "integrated gradient noise":
        print('Performing Integrated Gradients Noise Tunnel Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model, transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(image, n_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            cmap=linear_seg_color_map(name, color, N),
            show_colorbar=True)



    # Check if mode is Gradient Shap
    elif interpret_type == "gradient shap":

        print('Performing Gradient Shap Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model,transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([image * 0, image * 1])

        attributions_gs = gradient_shap.attribute(image,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=linear_seg_color_map(name, color, N),
                                              show_colorbar=True)

        # Check if mode is Saliency
    elif interpret_type == "saliency":

        print('Performing Saliency Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model, transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
        saliency = Saliency(model)
        attributions_gs = saliency.attribute(image,target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=linear_seg_color_map(name, color, N),
                                              show_colorbar=True)



class LuekemiaNet(pl.LightningModule):
    def __init__(self, lr=0.01, weight_decay=5e-4):
        super(LuekemiaNet, self).__init__()

        self.save_hyperparameters()
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
            nn.Dropout(),
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
   
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    


image_path = '/home/allen/Drive C/Peter Moss AML Leukemia Research/Dataset/all_test/Im041_0.jpg'
label_idx = '/home/allen/Drive C/Peter Moss AML Leukemia Research/ALL-PyTorch-2020/Classifier/Model/class_idx.json'


seed_everything(SEED)
# train data directory
train_dir = '/home/allen/Drive C/Peter Moss AML Leukemia Research/Dataset/all_train/'
# train label directoy
train_csv = '/home/allen/Drive C/Peter Moss AML Leukemia Research/Dataset/train.csv'
# labels
class_name = ["zero", "one"]

# number of epoch
epochs = 20
# learning rate
learn_rate = 0.001
# read train CSV file

labels = pd.read_csv(train_csv)
# print label count
labels_count = labels.label.value_counts()
print(labels_count)
# print 5 label header
print(labels.head())
# splitting data into training and validation set
train, valid = train_test_split(labels, stratify = labels.label, test_size = 0.1, shuffle=True)
print(len(train),len(valid))
#data augmentation
training_transforms, validation_transforms = augmentation()

train_dataset = LeukemiaDataset(df_data=train, data_dir=train_dir, transform=training_transforms)
valid_dataset = LeukemiaDataset(df_data=valid, data_dir=train_dir, transform=validation_transforms)
train_sampler = SubsetRandomSampler(list(train.index))
valid_sampler = SubsetRandomSampler(list(valid.index))
# Prepare dataset for neural networks
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model_path = "weight2.pth"
model = LuekemiaNet()
early_stop_callback = EarlyStopping(
    monitor='val_acc',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='max'
)
trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=2, callbacks=[early_stop_callback])
trainer.fit(model, train_data_loader, valid_data_loader)
trainer.save_checkpoint(model_path)

real_model = model.load_from_checkpoint(model_path)

y_pred, y_test = get_predictions(real_model, valid_data_loader)
# Get model precision, recall and f1_score
print(classification_report(y_test, y_pred, target_names=class_name))
# Get model confusion matrix
cm = confusion_matrix(y_test, y_pred)
confusion_matrix2(cm, class_name,save_path='confusion_matrix.png')

#prediction = predict_probability(real_model, validation_transforms, image_path)
interpret_model(real_model, validation_transforms, image_path, label_idx, use_cpu=True, interpret_type="integrated gradients")
interpret_model(real_model, validation_transforms, image_path, label_idx, use_cpu=True, interpret_type="gradient shap")
interpret_model(real_model, validation_transforms, image_path, label_idx, use_cpu=True, interpret_type="saliency")


